@testset "IceSheetModel" begin
    grid = RectilinearGrid(arch; size = (8, 8), x = (0, 1e5), y = (0, 1e5), topology = (Bounded, Bounded, Flat))
    a = 1e-8
    m = 2e-9

    model = IceSheetModel(grid; rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                          friction = WeertmanFriction(grid; coefficient = 3e6),
                          surface_mass_balance = a, basal_melt_rate = m)
    set!(model; H = (x, y) -> 1000 + x / 1e3)

    @test model.grid === grid
    @test size(model.velocities.uσ) == (9, 8, 10)
    @test isnothing(model.thermodynamics)
    @test model.velocity_solver.iteration > 0
    @test maximum(abs, Array(interior(model.velocities.u))) > 0

    Δt = 1e5
    simulation = Simulation(model; Δt, stop_iteration = 3)
    run!(simulation)

    @test iteration(model) == 3
    @test model.clock.time ≈ 3Δt
    # The walls close the domain, so only the sources change the volume.
    H = Array(interior(model.geometry.H))
    x = xnodes(grid, Center())
    @test sum(H) ≈ sum(1000 .+ x ./ 1e3) * 8 + 64 * 3Δt * (a - m) rtol = 1e-13

    mktempdir() do dir
        simulation.output_writers[:checkpointer] = Checkpointer(model; schedule = IterationInterval(3), dir, prefix = "nunatak")
        simulation.stop_iteration = 6
        run!(simulation)
        Hsaved = Array(interior(model.geometry.H))

        restored = IceSheetModel(grid; rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                                 surface_mass_balance = a, basal_melt_rate = m)
        restored_simulation = Simulation(restored; Δt, stop_iteration = 6)
        restored_simulation.output_writers[:checkpointer] = Checkpointer(restored; schedule = IterationInterval(3), dir, prefix = "nunatak")
        set!(restored_simulation; checkpoint = :latest)

        @test iteration(restored) == 6
        @test Array(interior(restored.geometry.H)) == Hsaved
    end
end

function model_ice_stream_error(Nx; L = 100e3, H₀ = 500, δ = 20, A = 1e-24, stress_balance = DepthIntegrated())
    grid = RectilinearGrid(arch; size = (Nx, 4), x = (0, L), y = (0, L / Nx * 4), topology = (Bounded, Bounded, Flat))
    constants = IceSheetConstants(grid; ice_density = 917)
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000),
                                          picard_iteration_control = Convergence(; reltol = 1e-10, maxiter = 300))
    model = IceSheetModel(grid; Nσ = 4, constants, velocity_solver, stress_balance, rheology = GlenFlowLaw(grid; rate_factor = A),
                          bed = (x, y) -> δ * cos(π * x / L))
    set!(model; H = H₀)

    ρg = constants.ice_density * constants.gravitational_acceleration
    U = A * (ρg * δ / 2)^3 * L / π
    exact = XFaceField(grid)
    set!(exact, (x, y) -> U * (sin(π * x / L) - sin(π * x / L)^3 / 3))
    u = Array(interior(model.velocities.u))
    return maximum(abs, u .- Array(interior(exact))) / (2U / 3), model
end

@testset "IceSheetModel stress-balance update" begin
    error, model = model_ice_stream_error(64)
    @info "Model ice stream (n = 3) relative error" error iterations = model.velocity_solver.iteration
    @test error < 1e-3
    @test 0 < model.velocity_solver.iteration < 300

    (; H, s, φ, φᵘ) = model.geometry
    @test Array(interior(s)) ≈ Array(interior(model.bed)) .+ Array(interior(H))
    @test all(Array(interior(φ)) .== 1) && all(Array(interior(φᵘ)) .== 1)

    (; u, uᵇ, uσ) = model.velocities
    @test Array(interior(uᵇ)) == Array(interior(u))
    @test all(k -> Array(interior(uσ, :, :, k)) == Array(interior(u, :, :, 1)), 1:size(uσ, 3))

    grid = RectilinearGrid(arch; size = (4, 4), x = (0, 1e4), y = (0, 1e4), topology = (Bounded, Bounded, Flat))
    floating = IceSheetModel(grid; grounding = SubgridGroundedFraction(), bed = -2000)
    set!(floating; H = 100)
    ρᵢ_ρw = floating.constants.ice_density / floating.constants.ocean_density
    @test all(Array(interior(floating.geometry.s)) .≈ (1 - ρᵢ_ρw) * 100)
    @test all(Array(interior(floating.geometry.φ)) .== 0)
    @test all(Array(interior(floating.geometry.N)) .== 0)
    @test isnothing(floating.velocity_solver)

end

@testset "IceSheetModel DIVA update" begin
    # Without basal drag there is no vertical shear, so DIVA reduces to SSA.
    error, model = model_ice_stream_error(64; stress_balance = DepthIntegrated(shear = DIVAShear()))
    @test error < 1e-3
    (; u, uᵇ, uσ) = model.velocities
    @test Array(interior(uᵇ)) == Array(interior(u))
    @test all(k -> Array(interior(uσ, :, :, k)) ≈ Array(interior(u, :, :, 1)), 1:size(uσ, 3))

    grid = RectilinearGrid(arch; size = (32, 4), x = (0, 1e5), y = (0, 1.25e4), topology = (Bounded, Bounded, Flat))
    rheology = GlenFlowLaw(grid; rate_factor = 1e-24)
    friction = WeertmanFriction(grid; coefficient = 1e12, exponent = 1)
    ssa = IceSheetModel(grid; rheology, friction)
    diva = IceSheetModel(grid; rheology, friction, stress_balance = DepthIntegrated(shear = DIVAShear()))
    for model in (ssa, diva)
        set!(model; H = (x, y) -> 1000 + x / 1e2)
    end

    ū, uᵇ = Array(interior(diva.velocities.u)), Array(interior(diva.velocities.uᵇ))
    uσ = Array(interior(diva.velocities.uσ))
    Δζ = reshape(diff(sigma_faces(size(uσ, 3))), 1, 1, :)
    @test 0 < diva.velocity_solver.iteration < 100
    @test maximum(abs, ū) > 2 * maximum(abs, Array(interior(ssa.velocities.u)))
    @test all(abs.(uᵇ) .<= abs.(ū))
    @test maximum(abs, sum(uσ .* Δζ, dims = 3) .- ū) < 1e-2 * maximum(abs, ū)
end
