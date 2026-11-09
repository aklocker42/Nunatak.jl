using Nunatak: compute_basal_drag!, update_stress_balance_coefficients!, ssa_operator!

function uniform_drag(friction, grid, U; φ₀ = 1, N₀ = nothing)
    u, v = XFaceField(grid), YFaceField(grid)
    set!(u, U)
    φᵘ, φᵛ = XFaceField(grid), YFaceField(grid)
    set!(φᵘ, φ₀)
    set!(φᵛ, φ₀)
    N = isnothing(N₀) ? nothing : CenterField(grid)
    isnothing(N₀) || set!(N, N₀)
    fill_halo_regions!((u, v))
    isnothing(N) || fill_halo_regions!(N)
    βᵘ, βᵛ = XFaceField(grid), YFaceField(grid)
    compute_basal_drag!(βᵘ, βᵛ, friction, u, v, φᵘ, φᵛ, N)
    return Array(interior(βᵘ))
end

function sliding_slab_speed(friction, τ₀, N₀)
    grid = RectilinearGrid(arch; size = (4, 4), x = (0, 1e3), y = (0, 1e3), topology = (Periodic, Periodic, Flat))
    σgrid = SigmaGrid(grid; Nσ = 2)
    H, N = CenterField(grid), CenterField(grid)
    φᵘ, φᵛ = XFaceField(grid), YFaceField(grid)
    set!(H, 500)
    set!(φᵘ, 1)
    set!(φᵛ, 1)
    set!(N, N₀)
    fill_halo_regions!((H, N))
    τ = VelocityVector(grid)
    set!(τ.u, τ₀ * 250^2)

    η̄H, η̄Hᶠᶠ, F₂ = Field{Center, Center, Nothing}(grid), Field{Face, Face, Nothing}(grid), Field{Center, Center, Nothing}(grid)
    βᵘ, βᵛ = Field{Face, Center, Nothing}(grid), Field{Center, Face, Nothing}(grid)
    coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, CenterField(σgrid), F₂, H, GlenFlowLaw(grid; rate_factor = 1e-24), friction, φᵘ, φᵛ, N)

    solver = VelocityConjugateGradientSolver(ssa_operator!, grid; iteration_control = Convergence(; reltol = 1e-13, maxiter = 100))
    picard = PicardIteration(grid; iteration_control = Convergence(; reltol = 1e-12, maxiter = 300))
    x = VelocityVector(grid)
    iterations = picard_solve!(x, picard, solver, update_stress_balance_coefficients!, τ, coefficients...)
    return Array(interior(x.u)), iterations
end

function operator_diagonal(grid, coefficients)
    e, q = VelocityVector(grid), VelocityVector(grid)
    diagonal = VelocityVector(grid)
    for (name, field) in ((:u, diagonal.u), (:v, diagonal.v))
        Nx, Ny, _ = size(field)
        for j in 1:Ny, i in 1:Nx
            fill!(e, 0)
            getproperty(e, name)[i, j, 1] = 1
            fill_halo_regions!(e)
            ssa_operator!(q, e, coefficients...)
            field[i, j, 1] = getproperty(q, name)[i, j, 1]
        end
    end
    return diagonal
end

@testset "Friction and diagonal preconditioner" begin
    grid = RectilinearGrid(arch; size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat))
    m, C, U = 1/3, 3e4, 1e-5
    weertman = WeertmanFriction(grid; coefficient = C, exponent = m, velocity_regularization = 0)
    @test all(uniform_drag(weertman, grid, U) .≈ C * U^(m - 1))
    @test all(uniform_drag(weertman, grid, U; φ₀ = 0.5) .≈ C * U^(m - 1) / 2)

    α², N₀ = 0.5, 1e6
    coulomb = RegularizedCoulombFriction(grid; coefficient = C, coulomb_coefficient = α², exponent = m, velocity_regularization = 0)
    @test uniform_drag(coulomb, grid, 1e-12; N₀)[1] / (C * 1e-12^(m - 1)) ≈ 1 rtol = 1e-3
    @test uniform_drag(coulomb, grid, 1e8; N₀)[1] * 1e8 / (α² * N₀) ≈ 1 rtol = 1e-3

    τ₀ = 1e5
    u, iterations = sliding_slab_speed(WeertmanFriction(grid; coefficient = C, exponent = m, velocity_regularization = 1e-12), τ₀, N₀)
    @test iterations < 300
    @test all(u .≈ (τ₀ / C)^(1 / m))

    u, iterations = sliding_slab_speed(RegularizedCoulombFriction(grid; coefficient = C, coulomb_coefficient = α², exponent = m,
                                                                  velocity_regularization = 1e-12), τ₀, N₀)
    τᶜ = α² * N₀
    Uᵉ = (τ₀ * τᶜ)^(1 / m) / (C^(1 / m) * (τᶜ^(1 / m) - τ₀^(1 / m)))
    @info "Sliding slab Picard iterations" iterations
    @test iterations < 300
    @test all(u .≈ Uᵉ)

    Nx, Ny = 7, 6
    stretched_x = collect(range(0, 1, length = Nx + 1)) .^ 1.4
    probe_grid = RectilinearGrid(CPU(); size = (Nx, Ny), x = stretched_x, y = (0, 1), topology = (Bounded, Bounded, Flat))
    for walls in (FreeSlipWalls(), NoSlipWalls())
        coefficients = (ssa_coefficients(probe_grid)..., walls)
        exact_diagonal = operator_diagonal(probe_grid, coefficients)
        z, r = VelocityVector(probe_grid), VelocityVector(probe_grid)
        fill!(r, 1)
        precondition!(z, StressBalanceDiagonalPreconditioner(), r, coefficients...)
        @test interior(z.u)[2:Nx, :, 1] .* interior(exact_diagonal.u)[2:Nx, :, 1] ≈ ones(Nx - 1, Ny)
        @test interior(z.v)[:, 2:Ny, 1] .* interior(exact_diagonal.v)[:, 2:Ny, 1] ≈ ones(Nx, Ny - 1)
    end

    contrast_grid = RectilinearGrid(arch; size = (48, 32), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat))
    coefficients = ssa_coefficients(contrast_grid; η̄H = (x, y) -> 1 + 1e3 * x^4, β = (x, y) -> 1e4 * exp(-20x))
    b = VelocityVector(contrast_grid)
    set!(b.u, (x, y) -> sin(π * x) * cos(2π * y))
    set!(b.v, (x, y) -> x * sin(2π * y))
    control = Convergence(; reltol = 1e-10, maxiter = 10_000)
    plain = VelocityConjugateGradientSolver(ssa_operator!, contrast_grid; iteration_control = control)
    jacobi = VelocityConjugateGradientSolver(ssa_operator!, contrast_grid; iteration_control = control,
                                             preconditioner = StressBalanceDiagonalPreconditioner())
    x₁, x₂ = VelocityVector(contrast_grid), VelocityVector(contrast_grid)
    solve!(x₁, plain, b, coefficients...)
    solve!(x₂, jacobi, b, coefficients...)
    @info "CG iterations" plain = plain.iteration jacobi = jacobi.iteration
    @test jacobi.iteration < plain.iteration
    @test Array(interior(x₂.u)) ≈ Array(interior(x₁.u)) rtol = 1e-7
end
