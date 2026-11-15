# # Inverting for basal friction with Enzyme
#
# Ice-sheet models need the friction between the ice and its bed, which cannot be observed directly. It is
# usually inferred from surface velocities, which satellites measure, by an inversion: find the friction for which
# the modelled velocities match the observed ones. This example runs a twin experiment. We make synthetic
# observations from a known friction field, add noise, and recover the friction with gradients that Enzyme computes
# through the velocity solve. The example shows how to
#
#   * write a cost function that solves the stress balance,
#   * differentiate it with Enzyme's reverse mode,
#   * minimise it with L-BFGS from Optim.jl.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Enzyme, Optim, CairoMakie"
# pkg"add https://github.com/aklocker42/Nunatak.jl"
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!
using Nunatak
using Enzyme
using Optim: Optim, optimize, LBFGS
using CairoMakie
using Printf
using Random

year = years = 31556926seconds

# ## The ice stream
#
# A 1500 m thick ice stream thins toward a floating shelf at x = 140 km. Its bed slopes down to below sea level.

Lx, Ly = 160kilometers, 80kilometers
Nx, Ny = 40, 20

grid = RectilinearGrid(CPU(); size = (Nx, Ny), halo = (4, 4), x = (0, Lx), y = (0, Ly), topology = (Bounded, Bounded, Flat))

thickness(x, y) = ifelse(x < 140kilometers, 1500 - 7e-3 * x - 200 * ((y - Ly / 2) / (Ly / 2))^2, 0.0)

# The friction coefficient is written as ``C = C_0 \exp(γ)``, so that ``C`` stays positive whatever ``γ`` the
# optimiser tries. We invert for ``γ``.

C₀ = 3e6

# Picard iterations converge the nonlinear viscosity; we run a fixed number so that every cost evaluation does the
# same work.

velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000),
                                      picard_iteration_control = FixedIterations(30))

model = IceSheetModel(grid; velocity_solver,
                      rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                      friction = WeertmanFriction(grid; coefficient = C₀),
                      grounding = SubgridGroundedFraction(),
                      bed = (x, y) -> 100 - 1e-2 * x)

set!(model; H = thickness)

# ## Synthetic observations
#
# The true friction has two sticky patches and one slippery patch. We solve for the velocities it gives and add
# noise of 1% of their root-mean-square value.

bump(x, y, x₀, y₀, r) = exp(-((x - x₀)^2 + (y - y₀)^2) / r^2)

true_friction(x, y) = 0.8 * bump(x, y, 30kilometers, 25kilometers, 8kilometers) +
                      0.8 * bump(x, y, 50kilometers, 55kilometers, 8kilometers) -
                      1.0 * bump(x, y, 40kilometers, 40kilometers, 10kilometers)

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

set_friction!(model, γ_true)
solve_velocities!(model)

u_true = Array(interior(model.velocities.u, :, :, 1))
v_true = Array(interior(model.velocities.v, :, :, 1))
grounded = Array(interior(model.geometry.φ, :, :, 1)) .> 0.5

rng = Xoshiro(1)
noise = 0.01 * sqrt(sum(abs2, u_true) / length(u_true))
uᵒ = u_true .+ noise .* randn(rng, size(u_true))
vᵒ = v_true .+ noise .* randn(rng, size(v_true))
norm = sum(abs2, uᵒ) + sum(abs2, vᵒ)

@info @sprintf("%d of %d cells grounded, max |u| = %.1f m/a, noise σ = %.2f m/a",
               count(grounded), length(grounded), maximum(abs, u_true) * year, noise * year)

# ## The cost function
#
# The cost is the velocity misfit, which Enzyme differentiates with respect to the friction coefficient. Nunatak
# provides adjoint rules for the velocity solve, so the gradient costs about as much as a forward solve.

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

# Noise makes the inversion ill posed, so we add a penalty ``\frac{λ}{2} \sum |Δγ|^2`` on differences of ``γ``
# between neighbouring grounded cells, whose gradient we write by hand.

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

# The chain rule through ``C = C_0 \exp(γ)`` turns ``∂J/∂C`` into ``∂J/∂γ``.

function cost_and_gradient!(g, γ, λ)
    set_friction!(model, γ)
    dmodel = Enzyme.make_zero(model)
    _, J = autodiff(set_runtime_activity(ReverseWithPrimal), misfit, Active,
                    Duplicated(model, dmodel), Const(uᵒ), Const(vᵒ), Const(norm))
    ∂J_∂C = Array(interior(dmodel.friction.coefficient, :, :, 1))
    g .= ∂J_∂C .* C₀ .* exp.(γ)
    return J + regularization!(g, γ, grounded, λ)
end

# ## Running the inversion
#
# We start from ``γ = 0`` and run L-BFGS for two regularization strengths.

function invert_friction(λ)
    history = Float64[]

    function objective(F, G, γ)
        J = cost_and_gradient!(G === nothing ? similar(γ) : G, γ, λ)
        push!(history, J)
        return J
    end

    result = optimize(Optim.NLSolversBase.only_fg!(objective), zeros(Nx, Ny), LBFGS(),
                      Optim.Options(iterations = 100, g_tol = 1e-10, show_trace = true, show_every = 5))

    γ = Optim.minimizer(result)
    set_friction!(model, γ)
    solve_velocities!(model)
    u = Array(interior(model.velocities.u, :, :, 1))
    v = Array(interior(model.velocities.v, :, :, 1))

    return (; λ, γ, u, v, history)
end

inversions = [invert_friction(λ) for λ in (1e-3, 1e-5)]

for r in inversions
    γ_error = sqrt(sum(abs2, (r.γ .- γ_true)[grounded]) / count(grounded))
    @info @sprintf("λ = %.0e: %d cost evaluations, rms γ error on grounded ice %.3f", r.λ, length(r.history), γ_error)
end

# ## Plotting
#
# The stronger regularization, ``λ = 10^{-3}``, smooths the patches away; ``λ = 10^{-5}`` recovers them and fits
# the velocities down to the noise. Only grounded ice, inside the black contour, constrains the friction.

masked(a) = ifelse.(grounded, a, NaN)
centred_speed(u, v) = sqrt.(((u[1:end-1, :] .+ u[2:end, :]) ./ 2) .^ 2 .+ ((v[:, 1:end-1] .+ v[:, 2:end]) ./ 2) .^ 2)

x, y = xᶜ ./ kilometer, yᶜ ./ kilometer

fig = Figure(size = (1400, 800))

panels = [(γ_true, "true γ"), (inversions[1].γ, "inverted γ, λ = 1e-3"), (inversions[2].γ, "inverted γ, λ = 1e-5")]
for (n, (γ, title)) in enumerate(panels)
    ax = Axis(fig[1, n]; title, xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, x, y, masked(γ); colormap = :balance, colorrange = (-1, 1))
    contour!(ax, x, y, Float64.(grounded); levels = [0.5], color = :black)
    n == 3 && Colorbar(fig[1, 4], hm; label = "γ = log(C / C₀)")
end

ax = Axis(fig[2, 1]; title = "observed speed", xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
hm = heatmap!(ax, x, y, log10.(max.(centred_speed(uᵒ, vᵒ) .* year, 1)); colormap = :viridis)
Colorbar(fig[3, 1], hm; label = "log₁₀ speed (m/a)", vertical = false, flipaxis = false)

r = inversions[2]
ax = Axis(fig[2, 2]; title = "inverted − observed speed (m/a), λ = 1e-5", xlabel = "x (km)", aspect = DataAspect())
hm = heatmap!(ax, x, y, (centred_speed(r.u, r.v) .- centred_speed(uᵒ, vᵒ)) .* year; colormap = :balance,
              colorrange = (-300, 300))
Colorbar(fig[3, 2], hm; vertical = false, flipaxis = false)

ax = Axis(fig[2, 3]; title = "cost during L-BFGS", xlabel = "evaluation", ylabel = "J", yscale = log10)
for r in inversions
    lines!(ax, r.history; label = @sprintf("λ = %.0e", r.λ))
end
axislegend(ax)

save("friction_inversion.png", fig)
