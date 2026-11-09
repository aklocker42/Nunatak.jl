using Nunatak: ssa_operator!, compute_viscosity!
using Oceananigans.Operators: Azᶜᶜᶜ

function ssa_coefficients(grid; η̄H = (x, y) -> 1 + x^2 + sin(2π * y)^2 / 2,
                                β = (x, y) -> 1 + x * y)
    η̄Hᶜᶜ = Field{Center, Center, Nothing}(grid)
    η̄Hᶠᶠ = Field{Face, Face, Nothing}(grid)
    βᵘ = Field{Face, Center, Nothing}(grid)
    βᵛ = Field{Center, Face, Nothing}(grid)
    set!(η̄Hᶜᶜ, η̄H)
    set!(η̄Hᶠᶠ, η̄H)
    set!(βᵘ, β)
    set!(βᵛ, β)
    fill_halo_regions!((η̄Hᶜᶜ, η̄Hᶠᶠ, βᵘ, βᵛ))
    return η̄Hᶜᶜ, η̄Hᶠᶠ, βᵘ, βᵛ
end

function apply_ssa_operator(p, coefficients)
    q = VelocityVector(p.u.grid)
    fill_halo_regions!(p)
    ssa_operator!(q, p, coefficients...)
    return q
end

function manufactured_ssa_error(N)
    grid = RectilinearGrid(arch; size = (N, N), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat))
    η₀, β₀, a, b = 1, 1, 2π, 4π
    coefficients = (ssa_coefficients(grid; η̄H = η₀, β = β₀)..., FreeSlipWalls())

    uᵉ(x, y) = sin(a * x) * cos(b * y)
    vᵉ(x, y) = cos(a * x) * sin(b * y)
    A = 1 / N^2
    rhs = VelocityVector(grid)
    set!(rhs.u, (x, y) -> A * (η₀ * (4a^2 + b^2 + 3a * b) + β₀) * uᵉ(x, y))
    set!(rhs.v, (x, y) -> A * (η₀ * (4b^2 + a^2 + 3a * b) + β₀) * vᵉ(x, y))

    solver = VelocityConjugateGradientSolver(ssa_operator!, grid;
                                             iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000))
    x = VelocityVector(grid)
    solve!(x, solver, rhs, coefficients...)

    exact = VelocityVector(grid)
    set!(exact.u, uᵉ)
    set!(exact.v, vᵉ)
    return max(maximum(abs, Array(interior(x.u)) .- Array(interior(exact.u))),
               maximum(abs, Array(interior(x.v)) .- Array(interior(exact.v))))
end

@testset "SSA operator" begin
    Nx, Ny = 12, 10
    stretched_x = collect(range(0, 1, length = Nx + 1)) .^ 1.3
    grid = RectilinearGrid(arch; size = (Nx, Ny), x = stretched_x, y = (0, 1), topology = (Bounded, Periodic, Flat))
    coefficients = (ssa_coefficients(grid)..., FreeSlipWalls())

    p = VelocityVector(grid)
    set!(p.u, (x, y) -> sin(π * x) * (cos(2π * y) + 2))
    set!(p.v, (x, y) -> cos(2x) + sin(4π * y) * x)
    y = VelocityVector(grid)
    set!(y.u, (x, y) -> x^2 - cos(2π * y))
    set!(y.v, (x, y) -> exp(-x) * sin(2π * y))

    Ap = apply_ssa_operator(p, coefficients)
    Ay = apply_ssa_operator(y, coefficients)
    @info "SSA operator symmetry" dot(Ap, y) dot(p, Ay)
    @test dot(Ap, y) ≈ dot(p, Ay)
    @test dot(Ap, p) > 0
    @test dot(Ay, y) > 0

    solver = VelocityConjugateGradientSolver(ssa_operator!, grid;
                                             iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000))
    x = VelocityVector(grid)
    solve!(x, solver, Ap, coefficients...)
    @test Array(interior(x.u)) ≈ Array(interior(p.u)) rtol = 1e-8
    @test Array(interior(x.v)) ≈ Array(interior(p.v)) rtol = 1e-8
    @test all(iszero, Array(interior(x.u))[[1, Nx + 1], :, :])

    errors = [manufactured_ssa_error(N) for N in (32, 64)]
    @info "SSA manufactured-solution error" errors ratio = errors[1] / errors[2]
    @test errors[2] < 5e-3
    @test errors[1] / errors[2] > 3.5
end

@testset "Free-slip walls" begin
    Nx, Ny = 6, 5
    β₀, U = 2, 3
    for topology in ((Periodic, Bounded, Flat), (Bounded, Periodic, Flat))
        grid = RectilinearGrid(arch; size = (Nx, Ny), x = (0, 1), y = (0, 1), topology)
        coefficients = (ssa_coefficients(grid; β = β₀)..., FreeSlipWalls())
        p = VelocityVector(grid)
        along_wall = topology[1] === Periodic ? p.u : p.v
        set!(along_wall, U)
        Ap = apply_ssa_operator(p, coefficients)
        Ap_along = topology[1] === Periodic ? Ap.u : Ap.v
        @test Array(interior(Ap_along)) ≈ fill(β₀ * U / (Nx * Ny), size(interior(Ap_along)))
    end

    stretched_x = collect(range(0, 1, length = Nx + 1)) .^ 1.3
    grid = RectilinearGrid(arch; size = (Nx, Ny), x = stretched_x, y = (0, 1), topology = (Bounded, Bounded, Flat))
    coefficients = (ssa_coefficients(grid)..., FreeSlipWalls())
    p, y = VelocityVector(grid), VelocityVector(grid)
    set!(p.u, (x, y) -> sin(π * x) + y)
    set!(p.v, (x, y) -> x * y^2)
    set!(y.u, (x, y) -> cos(x) * y)
    set!(y.v, (x, y) -> sin(3y) + x)
    Ap = apply_ssa_operator(p, coefficients)
    Ay = apply_ssa_operator(y, coefficients)
    @test dot(Ap, y) ≈ dot(p, Ay)
    @test dot(Ap, p) > 0
end

# Channel flow between no-slip walls under a uniform body force f: u = f y (W - y) / 2η₀.
function channel_flow_error(N; along_x = true)
    W, η₀, f = 1, 2, 3
    topology = along_x ? (Periodic, Bounded, Flat) : (Bounded, Periodic, Flat)
    grid = RectilinearGrid(arch; size = along_x ? (4, N) : (N, 4), x = (0, along_x ? 1 : W), y = (0, along_x ? W : 1), topology)
    coefficients = (ssa_coefficients(grid; η̄H = η₀, β = 0)..., NoSlipWalls())
    rhs = VelocityVector(grid)
    set!(along_x ? rhs.u : rhs.v, f * Azᶜᶜᶜ(1, 1, 1, grid))
    solver = VelocityConjugateGradientSolver(ssa_operator!, grid; iteration_control = Convergence(; reltol = 1e-13, maxiter = 10_000),
                                             preconditioner = StressBalanceDiagonalPreconditioner())
    x = VelocityVector(grid)
    solve!(x, solver, rhs, coefficients...)
    exact = VelocityVector(grid)
    along_x ? set!(exact.u, (x, y) -> f * y * (W - y) / 2η₀) : set!(exact.v, (x, y) -> f * x * (W - x) / 2η₀)
    U = along_x ? x.u : x.v
    Uᵉ = along_x ? exact.u : exact.v
    across = along_x ? x.v : x.u
    return maximum(abs, Array(interior(U)) .- Array(interior(Uᵉ))) / maximum(abs, Array(interior(Uᵉ))), maximum(abs, Array(interior(across)))
end

@testset "No-slip walls" begin
    for along_x in (true, false)
        (e₁, a₁), (e₂, a₂) = channel_flow_error(16; along_x), channel_flow_error(32; along_x)
        @info "No-slip channel flow error" along_x e₁ e₂ ratio = e₁ / e₂
        @test e₂ < 2e-3
        @test e₁ / e₂ > 3.5
        @test max(a₁, a₂) < 1e-12
    end

    Nx, Ny = 6, 5
    stretched_x = collect(range(0, 1, length = Nx + 1)) .^ 1.3
    grid = RectilinearGrid(arch; size = (Nx, Ny), x = stretched_x, y = (0, 1), topology = (Bounded, Bounded, Flat))
    coefficients = (ssa_coefficients(grid)..., NoSlipWalls())
    p, y = VelocityVector(grid), VelocityVector(grid)
    set!(p.u, (x, y) -> sin(π * x) + y)
    set!(p.v, (x, y) -> x * y^2)
    set!(y.u, (x, y) -> cos(x) * y)
    set!(y.v, (x, y) -> sin(3y) + x)
    Ap = apply_ssa_operator(p, coefficients)
    Ay = apply_ssa_operator(y, coefficients)
    @test dot(Ap, y) ≈ dot(p, Ay)
    @test dot(Ap, p) > dot(p, apply_ssa_operator(p, (ssa_coefficients(grid)..., FreeSlipWalls())))

    channel = RectilinearGrid(arch; size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Bounded, Flat))
    σgrid = SigmaGrid(channel; Nσ = 2)
    u, v = XFaceField(channel), YFaceField(channel)
    set!(u, 1)
    fill_halo_regions!((u, v))
    free, noslip = CenterField(σgrid), CenterField(σgrid)
    compute_viscosity!(free, GlenFlowLaw(channel; rate_factor = 1, exponent = 3), u, v, FreeSlipWalls())
    compute_viscosity!(noslip, GlenFlowLaw(channel; rate_factor = 1, exponent = 3), u, v, NoSlipWalls())
    @test Array(interior(noslip))[:, 1, :] ≈ Array(interior(noslip))[:, 4, :]
    @test all(Array(interior(noslip))[:, 1, :] .< Array(interior(free))[:, 1, :])
    @test Array(interior(noslip))[:, 2:3, :] ≈ Array(interior(free))[:, 2:3, :]
end
