@testset "IceSheetModel" begin
    grid = RectilinearGrid(arch; size = (8, 8), x = (0, 1e5), y = (0, 1e5), topology = (Bounded, Bounded, Flat))
    a = 1e-8
    m = 2e-9

    model = IceSheetModel(grid; rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                          friction = WeertmanFriction(grid; coefficient = 1e4),
                          stress_balance = DepthIntegrated(shear = DIVAShear()),
                          surface_mass_balance = a, basal_melt_rate = m)
    set!(model; H = (x, y) -> 1000 + x / 1e3)

    @test model.grid === grid
    @test size(model.velocities.uσ) == (9, 8, 10)
    @test isnothing(model.thermodynamics)

    Δt = 1e5
    simulation = Simulation(model; Δt, stop_iteration = 3)
    run!(simulation)

    @test iteration(model) == 3
    @test model.clock.time ≈ 3Δt
    H = Array(interior(model.geometry.H))
    x = xnodes(grid, Center())
    @test H[:, 1, 1] ≈ 1000 .+ x ./ 1e3 .+ 3Δt * (a - m)

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
