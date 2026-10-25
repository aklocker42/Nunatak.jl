using Oceananigans.Advection: cell_advection_timescale

# Eastward flow between walls, with zero velocity on the wall faces.
function transport_model(; advection = UpwindBiased(order = 1), Nx = 64, L = 64e3, U = 1e-6, width = L / 16,
                         velocity = (x, y) -> U * sin(π * x / L),
                         background = 0, surface_mass_balance = 0, basal_melt_rate = 0)
    grid = RectilinearGrid(arch; size = (Nx, 4), halo = (4, 4), x = (0, L), y = (0, 4L / Nx), topology = (Bounded, Periodic, Flat))
    model = IceSheetModel(grid; velocity_solver = nothing, advection, surface_mass_balance, basal_melt_rate)
    set!(model; H = (x, y) -> background + 100 * exp(-((x - L / 4) / width)^2), u = velocity)
    return model
end

volume(model) = sum(Array(interior(model.geometry.H))) * model.grid.Δxᶜᵃᵃ * model.grid.Δyᵃᶜᵃ

function centre_of_mass(model)
    H = Array(interior(model.geometry.H, :, :, 1))
    x = Array(xnodes(model.grid, Center()))
    return sum(x .* H) / sum(H)
end

function run_steps!(model, Δt, Nt)
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

@testset "Thickness transport" begin
    schemes = (UpwindBiased(order = 1), UpwindBiased(order = 3), WENO(order = 5), WENO(order = 5, bounds = (0.0, Inf)))

    @testset "Volume is conserved by $(summary(advection))" for advection in schemes
        model = transport_model(; advection, background = 100)
        V₀ = volume(model)
        Δt = 0.5 * cell_advection_timescale(model)
        run_steps!(model, Δt, 32)
        @test abs(volume(model) - V₀) / V₀ < 1e-13
    end

    @testset "Sources change the volume by the integrated a − m" begin
        a, m = 3e-9, 1e-9
        model = transport_model(; background = 100, surface_mass_balance = a, basal_melt_rate = (x, y) -> 2m * x / 64e3)
        V₀ = volume(model)
        Δt = 0.5 * cell_advection_timescale(model)
        run_steps!(model, Δt, 16)
        area = 64e3 * 4e3
        @test volume(model) ≈ V₀ + 16Δt * (a - m) * area rtol = 1e-13
    end

    @testset "Upwind transport moves the centre of mass at the flow speed" begin
        # Uniform flow; the bump stays far enough from the walls that no mass reaches the wall cells.
        U, L = 1e-6, 128e3
        model = transport_model(; U, L, Nx = 128, width = 2e3, velocity = (x, y) -> ifelse(0 < x < L, U, zero(x)))
        X₀ = centre_of_mass(model)
        Δt = 0.5 * cell_advection_timescale(model)
        @test Δt ≈ 0.5 * 1e3 / U
        run_steps!(model, Δt, 16)
        @test centre_of_mass(model) ≈ X₀ + U * 16Δt rtol = 1e-12
    end

    @testset "Bounded WENO keeps a compact bump non-negative without clipping" begin
        # The bound holds up to a Courant number of 5/18; unbounded WENO undershoots and loses volume to the clip.
        drift = map((WENO(order = 5, bounds = (0.0, Inf)), WENO(order = 5))) do advection
            model = transport_model(; advection, width = 2e3)
            V₀ = volume(model)
            run_steps!(model, 0.25 * cell_advection_timescale(model), 64)
            @test minimum(Array(interior(model.geometry.H))) >= 0
            return abs(volume(model) - V₀) / V₀
        end
        @test drift[1] < 1e-12
        @test drift[2] > 1e-9

        small_halo = RectilinearGrid(arch; size = (8, 8), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat))
        @test_throws ArgumentError IceSheetModel(small_halo; velocity_solver = nothing, advection = WENO(order = 5, bounds = (0.0, Inf)))
    end

    @testset "TimeStepWizard follows the ice velocity" begin
        model = transport_model(; U = 2e-6)
        simulation = Simulation(model; Δt = 1.0, stop_iteration = 2)
        conjure_time_step_wizard!(simulation, IterationInterval(1); cfl = 0.5, max_change = Inf)
        run!(simulation)
        @test simulation.Δt ≈ 0.5 * cell_advection_timescale(model)
        @test simulation.Δt ≈ 0.5 * 1e3 / (2e-6 * sin(π * 32 / 64)) rtol = 1e-2
    end
end
