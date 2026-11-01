# Twin experiment: recover a basal friction field from noisy surface velocities, with C = C₀ exp(γ),
# cost = velocity misfit + Tikhonov on ∇γ, Enzyme gradients through the velocity solve, and L-BFGS.
# Writes friction_inversion_lambda*.jld2 to the current directory, one per regularisation strength.
# Usage: julia --project=path/to/Nunatak/examples -e 'using Pkg; Pkg.instantiate()'
#        julia --project=path/to/Nunatak/examples path/to/Nunatak/examples/friction_inversion.jl
using Oceananigans, Nunatak, Enzyme, JLD2, Printf, Random
using Optim: Optim, optimize, LBFGS
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!

Lx, Ly = 160e3, 80e3
Nx, Ny = 40, 20
C₀ = 3e6
noise = 0.01
sec_per_year = 3.15576e7

grid = RectilinearGrid(CPU(); size = (Nx, Ny), halo = (4, 4), x = (0, Lx), y = (0, Ly), topology = (Bounded, Bounded, Flat))
velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000),
                                      picard_iteration_control = FixedIterations(30))
model = IceSheetModel(grid; velocity_solver, rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                      friction = WeertmanFriction(grid; coefficient = C₀),
                      grounding = SubgridGroundedFraction(), bed = (x, y) -> 100 - 1e-2 * x)
thickness(x, y) = ifelse(x < 140e3, 1500 - 7e-3 * x - 200 * ((y - Ly / 2) / (Ly / 2))^2, 0.0)
set!(model; H = thickness)

bump(x, y, x₀, y₀, r) = exp(-((x - x₀)^2 + (y - y₀)^2) / r^2)
true_friction(x, y) = 0.8 * bump(x, y, 30e3, 25e3, 8e3) + 0.8 * bump(x, y, 50e3, 55e3, 8e3) - 1.0 * bump(x, y, 40e3, 40e3, 10e3)
xᶜ, yᶜ = xnodes(grid, Center()), ynodes(grid, Center())
γ_true = [true_friction(x, y) for x in xᶜ, y in yᶜ]

function set_friction!(model, γ)
    C = model.friction.coefficient
    interior(C)[:, :, 1] .= C₀ .* exp.(γ)
    fill_halo_regions!(C)
    return nothing
end

function solve_velocities!(model)
    fill!(model.velocities.u, 0)
    fill!(model.velocities.v, 0)
    update_state!(model)
    return nothing
end

function misfit(model, uᵒ, vᵒ, norm)
    fill_halo_regions!(model.friction.coefficient)
    fill!(model.velocities.u, 0)
    fill!(model.velocities.v, 0)
    update_state!(model)
    u, v = model.velocities.u, model.velocities.v
    J = zero(eltype(u))
    for j in 1:size(uᵒ, 2), i in 1:size(uᵒ, 1)
        @inbounds J += (u[i, j, 1] - uᵒ[i, j])^2
    end
    for j in 1:size(vᵒ, 2), i in 1:size(vᵒ, 1)
        @inbounds J += (v[i, j, 1] - vᵒ[i, j])^2
    end
    return J / 2norm
end

# Tikhonov ½ λ Σ |Δγ|² over neighbouring grounded cells, and its gradient.
function regularization!(g, γ, grounded, λ)
    Nx, Ny = size(γ)
    R = zero(eltype(γ))
    for j in 1:Ny, i in 1:Nx-1
        grounded[i, j] & grounded[i+1, j] || continue
        δ = γ[i+1, j] - γ[i, j]
        R += λ * δ^2 / 2
        g[i+1, j] += λ * δ
        g[i, j] -= λ * δ
    end
    for j in 1:Ny-1, i in 1:Nx
        grounded[i, j] & grounded[i, j+1] || continue
        δ = γ[i, j+1] - γ[i, j]
        R += λ * δ^2 / 2
        g[i, j+1] += λ * δ
        g[i, j] -= λ * δ
    end
    return R
end

set_friction!(model, γ_true)
solve_velocities!(model)
u_true = Array(interior(model.velocities.u, :, :, 1))
v_true = Array(interior(model.velocities.v, :, :, 1))
grounded = Array(interior(model.geometry.φ, :, :, 1)) .> 0.5
rng = Xoshiro(1)
scale = noise * sqrt(sum(abs2, u_true) / length(u_true))
uᵒ = u_true .+ scale .* randn(rng, size(u_true))
vᵒ = v_true .+ scale .* randn(rng, size(v_true))
norm = sum(abs2, uᵒ) + sum(abs2, vᵒ)
@printf("grounded cells %d / %d, max |u| = %.1f m/a, noise σ = %.2f m/a\n",
        count(grounded), length(grounded), maximum(abs, u_true) * sec_per_year, scale * sec_per_year)

function cost_and_gradient!(g, γ, λ)
    set_friction!(model, γ)
    dmodel = Enzyme.make_zero(model)
    _, J = autodiff(set_runtime_activity(ReverseWithPrimal), misfit, Active,
                    Duplicated(model, dmodel), Const(uᵒ), Const(vᵒ), Const(norm))
    ∂J_∂C = Array(interior(dmodel.friction.coefficient, :, :, 1))
    g .= ∂J_∂C .* C₀ .* exp.(γ)
    return J + regularization!(g, γ, grounded, λ)
end

function cost(γ, λ)
    set_friction!(model, γ)
    return misfit(model, uᵒ, vᵒ, norm) + regularization!(zeros(size(γ)), γ, grounded, λ)
end

function check_gradient(γ, λ; h = 1e-4)
    g = similar(γ)
    cost_and_gradient!(g, γ, λ)
    for index in ((8, 6), (12, 14), (30, 10))
        γ⁺, γ⁻ = copy(γ), copy(γ)
        γ⁺[index...] += h
        γ⁻[index...] -= h
        fd = (cost(γ⁺, λ) - cost(γ⁻, λ)) / 2h
        @printf("gradient check %-8s ad = % .6e  fd = % .6e  rel = %.1e\n", string(index), g[index...], fd, abs(g[index...] - fd) / abs(fd))
    end
    return nothing
end

rms(a) = sqrt(sum(abs2, a) / length(a))

function invert_friction(λ)
    γ₀ = zeros(Nx, Ny)
    @printf("λ = %.0e\n", λ)
    check_gradient(γ₀, λ)

    history = Float64[]

    function objective(F, G, γ)
        J = cost_and_gradient!(G === nothing ? similar(γ) : G, γ, λ)
        push!(history, J)
        return J
    end

    wall = @elapsed result = optimize(Optim.NLSolversBase.only_fg!(objective), γ₀, LBFGS(),
                                      Optim.Options(iterations = 100, g_tol = 1e-10, show_trace = true, show_every = 5))
    γ_inv = Optim.minimizer(result)
    println(result)

    set_friction!(model, γ_inv)
    solve_velocities!(model)
    u_inv = Array(interior(model.velocities.u, :, :, 1))
    v_inv = Array(interior(model.velocities.v, :, :, 1))
    @printf("wall %.0f s; misfit rms %.3f m/a (noise %.3f); γ error rms grounded %.3f (prior %.3f)\n",
            wall, rms(vcat(vec(u_inv .- uᵒ), vec(v_inv .- vᵒ))) * sec_per_year, scale * sec_per_year,
            rms((γ_inv .- γ_true)[grounded]), rms(γ_true[grounded]))

    jldsave(@sprintf("friction_inversion_lambda%.0e.jld2", λ); xᶜ = collect(xᶜ), yᶜ = collect(yᶜ), γ_true, γ_inv, grounded,
            u_true, v_true, uᵒ, vᵒ, u_inv, v_inv, history, thickness = Array(interior(model.geometry.H, :, :, 1)), λ, noise)

    return nothing
end

# λ = 1e-3 over-smooths the patches; λ = 1e-5 fits the velocities down to the noise.
for λ in (1e-3, 1e-5)
    invert_friction(λ)
end
