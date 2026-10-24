using Nunatak: compute_viscosity!, compute_driving_stress!, update_stress_balance_coefficients!, ssa_operator!

function uniform_strain_viscosity(grid, uᵉ)
    σgrid = SigmaGrid(grid; Nσ = 3)
    ησ = CenterField(σgrid)
    x = VelocityVector(grid)
    set!(x.u, uᵉ)
    fill_halo_regions!(x)
    rheology = GlenFlowLaw(grid; rate_factor = 1, strain_rate_regularization = 0)
    compute_viscosity!(ησ, rheology, x.u, x.v)
    return Array(interior(ησ))
end

function ice_stream_error(Nx; L = 100e3, H₀ = 500, δ = 20, ρ = 917, g = 9.81, A = 1e-24)
    grid = RectilinearGrid(arch; size = (Nx, 4), x = (0, L), y = (0, L / Nx * 4), topology = (Bounded, Periodic, Flat))
    σgrid = SigmaGrid(grid; Nσ = 4)

    H = CenterField(grid)
    s = CenterField(grid)
    set!(H, H₀)
    set!(s, (x, y) -> H₀ + δ * cos(π * x / L))
    fill_halo_regions!((H, s))

    τ = VelocityVector(grid)
    compute_driving_stress!(τ, H, s, ρ, g)

    rheology = GlenFlowLaw(grid; rate_factor = A)
    η̄H, η̄Hᶠᶠ, F₂ = Field{Center, Center, Nothing}(grid), Field{Face, Face, Nothing}(grid), Field{Center, Center, Nothing}(grid)
    βᵘ, βᵛ = Field{Face, Center, Nothing}(grid), Field{Center, Face, Nothing}(grid)
    ησ = CenterField(σgrid)
    coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, nothing, nothing, nothing)

    solver = VelocityConjugateGradientSolver(ssa_operator!, grid;
                                             iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000))
    picard = PicardIteration(grid; iteration_control = Convergence(; reltol = 1e-10, maxiter = 300))
    x = VelocityVector(grid)
    iterations = picard_solve!(x, picard, solver, update_stress_balance_coefficients!, τ, coefficients...)

    U = A * (ρ * g * δ / 2)^3 * L / π
    uᵉ(x, y) = U * (sin(π * x / L) - sin(π * x / L)^3 / 3)
    exact = XFaceField(grid)
    set!(exact, uᵉ)
    error = maximum(abs, Array(interior(x.u)) .- Array(interior(exact))) / (2U / 3)
    return error, iterations, maximum(abs, Array(interior(x.v))) / (2U / 3)
end

@testset "Glen viscosity and driving stress" begin
    a = 2.0
    ησ = uniform_strain_viscosity(RectilinearGrid(arch; size = (8, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat)),
                                  (x, y) -> a * x)
    @test all(ησ[2:7, :, :] .≈ a^(-2 / 3) / 2)

    ησ = uniform_strain_viscosity(RectilinearGrid(arch; size = (4, 8), x = (0, 1), y = (0, 1), topology = (Periodic, Bounded, Flat)),
                                  (x, y) -> a * y)
    @test all(ησ[:, 2:7, :] .≈ (a / 2)^(-2 / 3) / 2)

    grid = RectilinearGrid(arch; size = (6, 5), x = (0, 3), y = (0, 1), topology = (Bounded, Bounded, Flat))
    H, s = CenterField(grid), CenterField(grid)
    set!(H, (x, y) -> 1 + x)
    set!(s, (x, y) -> 2x + 3y)
    fill_halo_regions!((H, s))
    τ = VelocityVector(grid)
    compute_driving_stress!(τ, H, s, 2, 5)
    Az = 0.5 * 0.2
    xᶠ = 0.5:0.5:2.5
    yᶠ = 0.2:0.2:0.8
    xᶜ = 0.25:0.5:2.75
    τu, τv = Array(interior(τ.u)), Array(interior(τ.v))
    @test τu[2:6, :, 1] ≈ repeat(-Az * 10 .* (1 .+ xᶠ) .* 2, 1, 5)
    @test τv[:, 2:5, 1] ≈ repeat(-Az * 10 .* (1 .+ xᶜ) .* 3, 1, 4)
    @test all(iszero, τu[[1, 7], :, 1])
    @test all(iszero, τv[:, [1, 6], 1])

    errors_and_iterations = [ice_stream_error(N) for N in (32, 64)]
    errors = first.(errors_and_iterations)
    @info "Ice stream (n = 3) relative errors" errors iterations = getindex.(errors_and_iterations, 2) v = getindex.(errors_and_iterations, 3)
    @test all(e -> e[2] < 300, errors_and_iterations)
    @test all(e -> e[3] < 1e-10, errors_and_iterations)
    @test errors[2] < 1e-3
    @test errors[1] / errors[2] > 3.5
end
