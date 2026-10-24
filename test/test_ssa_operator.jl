using Nunatak: ssa_operator!

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
    coefficients = ssa_coefficients(grid; η̄H = η₀, β = β₀)

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
    coefficients = ssa_coefficients(grid)

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
        coefficients = ssa_coefficients(grid; β = β₀)
        p = VelocityVector(grid)
        along_wall = topology[1] === Periodic ? p.u : p.v
        set!(along_wall, U)
        Ap = apply_ssa_operator(p, coefficients)
        Ap_along = topology[1] === Periodic ? Ap.u : Ap.v
        @test Array(interior(Ap_along)) ≈ fill(β₀ * U / (Nx * Ny), size(interior(Ap_along)))
    end

    stretched_x = collect(range(0, 1, length = Nx + 1)) .^ 1.3
    grid = RectilinearGrid(arch; size = (Nx, Ny), x = stretched_x, y = (0, 1), topology = (Bounded, Bounded, Flat))
    coefficients = ssa_coefficients(grid)
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
