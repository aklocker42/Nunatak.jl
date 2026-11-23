using Nunatak: compute_effective_pressure!

function till_effective_pressure(law, P₀, Hw)
    (; reference_effective_pressure, overburden_fraction, reference_void_ratio, compressibility, maximum_water_thickness) = law
    s = clamp(Hw / maximum_water_thickness, 0, 1)
    Nᵗ = reference_effective_pressure * (overburden_fraction * P₀ / reference_effective_pressure)^s *
         10^(reference_void_ratio / compressibility * (1 - s))
    return min(Nᵗ, P₀)
end

function till_column_model(effective_pressure; H₀ = 1000, Hw = (1.5, 3))
    grid = RectilinearGrid(arch; size = (2, 1), x = (0, 2e3), y = (0, 1e3), topology = (Periodic, Periodic, Flat))
    thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux = 0, effective_pressure = effective_pressure(grid))
    model = IceSheetModel(grid; Nσ = 10, thermodynamics, constants = IceSheetConstants(grid; ice_density = 910),
                          velocity_solver = nothing, advection = UpwindBiased(order = 1))
    set!(model; H = H₀)
    set!(model.thermodynamics.state.Hw, (x, y) -> ifelse(x < 1e3, Hw[1], Hw[2]))
    return model
end

@testset "Till effective pressure" begin
    @testset "Dry, partly wet, saturated and floating till" begin
        grid = RectilinearGrid(arch; size = (5, 1), x = (0, 5), y = (0, 1), topology = (Bounded, Periodic, Flat))
        constants = IceSheetConstants(grid)
        law = TillEffectivePressure(grid)
        H, b, N = CenterField(grid), CenterField(grid), CenterField(grid)
        Hw = Field{Center, Center, Nothing}(grid)
        H₀, Hws = 1000, [0, 1.5, 2, 3, 1]
        set!(H, H₀)
        set!(b, (x, y) -> ifelse(x < 4, 0, -2000))
        set!(Hw, (x, y) -> Hws[clamp(ceil(Int, x), 1, 5)])
        compute_effective_pressure!(N, H, b, constants, law, Hw)
        P₀ = constants.ice_density * constants.gravitational_acceleration * H₀
        Nc = Array(interior(N, :, 1, 1))
        @test Nc[1] ≈ P₀
        @test Nc[2] ≈ till_effective_pressure(law, P₀, 1.5) < P₀
        @test Nc[3] ≈ law.overburden_fraction * P₀
        @test Nc[4] ≈ law.overburden_fraction * P₀
        @test Nc[5] == 0
        @test Nc[1] > Nc[2] > Nc[3]
    end

    @testset "Ice-free cells have zero effective pressure" begin
        grid = RectilinearGrid(arch; size = (2, 1), x = (0, 2), y = (0, 1), topology = (Bounded, Periodic, Flat))
        H, b, N = CenterField(grid), CenterField(grid), CenterField(grid)
        Hw = Field{Center, Center, Nothing}(grid)
        set!(H, (x, y) -> ifelse(x < 1, 0, 100))
        set!(Hw, 1)
        compute_effective_pressure!(N, H, b, IceSheetConstants(grid), TillEffectivePressure(grid), Hw)
        Nc = Array(interior(N, :, 1, 1))
        @test Nc[1] == 0
        @test Nc[2] > 0
    end

    @testset "The till drains and is capped, and the model sees the till effective pressure" begin
        Δt = 365.25 * 86400
        reference = till_column_model(grid -> nothing)
        till = till_column_model(TillEffectivePressure)
        law = till.thermodynamics.effective_pressure
        time_step!(reference, Δt)
        time_step!(till, Δt)
        Hwʳ = Array(interior(reference.thermodynamics.state.Hw))[:, 1, 1]
        Hwᵗ = Array(interior(till.thermodynamics.state.Hw))[:, 1, 1]
        @test Hwᵗ[1] ≈ Hwʳ[1] - Δt * law.drainage_rate rtol = 1e-12
        @test Hwᵗ[2] == law.maximum_water_thickness
        @test Hwʳ[2] > law.maximum_water_thickness
        @test Array(interior(till.thermodynamics.state.mᵍ)) == Array(interior(reference.thermodynamics.state.mᵍ))

        constants = till.constants
        P₀ = constants.ice_density * constants.gravitational_acceleration * Array(interior(till.geometry.H))[:, 1, 1]
        N = Array(interior(till.geometry.N))[:, 1, 1]
        # The geometry, and so `N`, is diagnosed before the energy step updates `Hw`.
        @test N ≈ till_effective_pressure.(Ref(law), P₀, [1.5, 3])
        Oceananigans.TimeSteppers.update_state!(till)
        @test Array(interior(till.geometry.N))[:, 1, 1] ≈ till_effective_pressure.(Ref(law), P₀, Hwᵗ)
        Nʳ = Array(interior(reference.geometry.N))[:, 1, 1]
        @test Nʳ ≈ P₀
    end
end
