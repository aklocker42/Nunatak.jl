# Reactant raises KernelAbstractions kernels through CUDA.jl, whatever the target backend.
using CUDA
using Reactant
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: update_state!

function reactant_test_shelf(arch)
    grid = RectilinearGrid(arch; size = (16, 8), halo = (4, 4), x = (0, 64e3), y = (0, 32e3),
                           topology = (Bounded, Bounded, Flat))
    # Converged linear solves, so the time step does not amplify round-off differences between the two.
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = FixedIterations(200),
                                          picard_iteration_control = FixedIterations(3))
    model = IceSheetModel(grid; Nσ = 4, velocity_solver,
                          rheology = GlenFlowLaw(grid; rate_factor = 1e-24, exponent = 3),
                          friction = WeertmanFriction(grid; coefficient = 1e6, exponent = 1),
                          bed = (x, y) -> -200 - 2e-3 * x,
                          stress_balance = DepthIntegrated(side_walls = NoSlipWalls()))
    set!(model.geometry.H, (x, y) -> 400 - 3e-3 * x)
    return model
end

relative_difference(a, b) = maximum(abs, Array(interior(a)) - Array(interior(b))) / maximum(abs, Array(interior(a)))

@testset "Reactant" begin
    cpu = reactant_test_shelf(CPU())
    traced = reactant_test_shelf(ReactantState())

    update! = @compile raise = true update_state!(traced)
    update!(traced)
    update_state!(cpu)
    @test relative_difference(cpu.velocities.u, traced.velocities.u) < 1e-10
    @test relative_difference(cpu.velocities.v, traced.velocities.v) < 1e-10

    Δt = 3e6
    step! = @compile raise = true time_step!(traced, Δt)
    # Reactant skips the extra update_state! that the first CPU time_step! does at iteration 0.
    cpu.clock.iteration = 1
    step!(traced, Δt)
    time_step!(cpu, Δt)
    @test relative_difference(cpu.geometry.H, traced.geometry.H) < 1e-10
    @test relative_difference(cpu.velocities.u, traced.velocities.u) < 1e-10
end
