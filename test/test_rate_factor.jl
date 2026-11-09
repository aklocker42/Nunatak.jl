using Nunatak: arrhenius_rate_factor, compute_viscosity!, compute_column_integrals!, pressure_melting_enthalpy,
               ice_sheet_velocities, ice_sheet_auxiliary_fields, update_diva_coefficients!, compute_velocity_profile!

interior_array(field) = Array(interior(field))

# Paterson-Budd values from the constants in PISM pism_config.cdl.
cold_branch(T′) = 3.61e-13 * exp(-6.0e4 / (8.31441 * T′))
warm_branch(T′) = 1.73e3 * exp(-13.9e4 / (8.31441 * T′))

# As slab_profile_error, with A(ζ) = A₀ (1 + 3 (1 - ζ)). Integrating 2A (τ (1 - ζ))³ H gives u(ζ) = uᵇ + U (17/20 - s⁴/4 - 3s⁵/5), s = 1 - ζ.
function depth_varying_slab_error(Nσ; H₀ = 1000, τ₀ = 1e4, A₀ = 1e-21, C = 1e10)
    grid = RectilinearGrid(arch; size = (4, 4), x = (0, 4e3), y = (0, 4e3), topology = (Periodic, Periodic, Flat))
    σgrid = SigmaGrid(grid; Nσ)
    velocities = ice_sheet_velocities(grid, σgrid)
    (; ησ, η̄H, η̄Hᶠᶠ, F₂, βᵘ, βᵛ, βᵇᵘ, βᵇᵛ) = ice_sheet_auxiliary_fields(grid, σgrid)

    H = CenterField(grid)
    set!(H, H₀)
    fill_halo_regions!(H)
    φᵘ, φᵛ = XFaceField(grid), YFaceField(grid)
    set!(φᵘ, 1)
    set!(φᵛ, 1)

    τ = VelocityVector(grid)
    set!(τ.u, τ₀ * 1e3^2)

    A = CenterField(σgrid)
    set!(A, (x, y, ζ) -> A₀ * (1 + 3 * (1 - ζ)))
    rheology = GlenFlowLaw(grid; rate_factor = A, strain_rate_regularization = 1e-16)
    friction = WeertmanFriction(grid; coefficient = C, exponent = 1)
    coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, φᵘ, φᵛ, nothing, nothing, βᵇᵘ, βᵇᵛ, velocities.uᵇ, velocities.vᵇ, FreeSlipWalls())

    solver = VelocityConjugateGradientSolver(ssa_operator!, grid; iteration_control = Convergence(; reltol = 1e-12, maxiter = 100))
    picard = PicardIteration(grid; iteration_control = Convergence(; reltol = 1e-12, maxiter = 300))
    x = VelocityVector(velocities.u, velocities.v)
    iterations = picard_solve!(x, picard, solver, update_diva_coefficients!, τ, coefficients...)
    fill_halo_regions!(ησ)
    compute_velocity_profile!(velocities, βᵇᵘ, βᵇᵛ, F₂, ησ, H)

    uᵇ = τ₀ / C
    U = 2A₀ * τ₀^3 * H₀
    s = 1 .- Array(znodes(σgrid, Center()))
    exact = uᵇ .+ U .* (17 / 20 .- s .^ 4 ./ 4 .- 3 .* s .^ 5 ./ 5)
    uσ = Array(interior(velocities.uσ, 1, 1, :))
    ū = interior_array(velocities.u)

    return (; iterations,
              profile = maximum(abs, uσ .- exact) / U,
              mean = maximum(abs, ū .- (uᵇ + 7U / 10)) / (7U / 10))
end

function rate_factor_model(; stress_balance = DepthIntegrated(), clausius_clapeyron = 0, rate_factor = grid -> ArrheniusRateFactor(grid),
                           surface_temperature = 243.15, geothermal_flux = 0)
    grid = RectilinearGrid(arch; size = (16, 4), halo = (4, 4), x = (0, 1.6e5), y = (0, 4e4), topology = (Bounded, Bounded, Flat))
    rheology = GlenFlowLaw(grid; rate_factor = rate_factor(grid), strain_rate_regularization = 1e-16)
    friction = WeertmanFriction(grid; coefficient = 1e10, exponent = 1)
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000),
                                          picard_iteration_control = Convergence(; reltol = 1e-11, maxiter = 500))
    thermodynamics = EnthalpyThermodynamics(grid; clausius_clapeyron, surface_temperature, geothermal_flux)
    return IceSheetModel(grid; stress_balance, rheology, friction, velocity_solver, thermodynamics, advection = UpwindBiased(order = 1),
                         constants = IceSheetConstants(grid; ice_density = 910))
end

sloped_thickness(x, y) = 1000 + x / 1e2

@testset "Arrhenius rate factor" begin
    grid = RectilinearGrid(arch; size = (1, 1), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat))
    parameters = ArrheniusRateFactor(grid).parameters

    @testset "Values" begin
        for T′ in (223.15, 243.15, 262.6)
            @test arrhenius_rate_factor(T′, 0.0, parameters) ≈ cold_branch(T′) rtol = 1e-13
        end
        for T′ in (263.7, 268.15, 273.15)
            @test arrhenius_rate_factor(T′, 0.0, parameters) ≈ warm_branch(T′) rtol = 1e-13
        end
        @test arrhenius_rate_factor(273.15, 0.005, parameters) ≈ warm_branch(273.15) * (1 + 181.25 * 0.005) rtol = 1e-13
        @test arrhenius_rate_factor(273.15, 0.05, parameters) ≈ warm_branch(273.15) * (1 + 181.25 * 0.01) rtol = 1e-13
    end

    @testset "Smooth transition" begin
        T′ = range(262, 264; length = 2001)
        A = [arrhenius_rate_factor(t, 0.0, parameters) for t in T′]
        @test all(diff(A) .> 0)
        slope = diff(log.(A)) ./ step(T′)
        @test maximum(abs, diff(slope)) * 2000 / 2 < 1
        # Half way between the branches at the transition temperature.
        @test arrhenius_rate_factor(263.15, 0.0, parameters) ≈ sqrt(cold_branch(263.15) * warm_branch(263.15)) rtol = 1e-13
    end

    @testset "A uniform 3-D field equals a 2-D field" begin
        grid = RectilinearGrid(arch; size = (8, 8), halo = (4, 4), x = (0, 1e5), y = (0, 1e5), topology = (Periodic, Periodic, Flat))
        σgrid = SigmaGrid(grid; Nσ = 4)
        u, v, uᵇ, vᵇ, βᵘ, βᵛ = XFaceField(grid), YFaceField(grid), XFaceField(grid), YFaceField(grid), XFaceField(grid), YFaceField(grid)
        set!(u, (x, y) -> 1e-6 * sin(2π * x / 1e5))
        set!(v, (x, y) -> 2e-6 * cos(2π * y / 1e5))
        set!(uᵇ, 1e-7)
        set!(vᵇ, 1e-7)
        set!(βᵘ, 1e10)
        set!(βᵛ, 1e10)
        fill_halo_regions!((u, v, uᵇ, vᵇ, βᵘ, βᵛ))
        A³ᴰ = CenterField(σgrid)
        set!(A³ᴰ, 1e-24)
        two_dimensional = GlenFlowLaw(grid; rate_factor = 1e-24)
        three_dimensional = GlenFlowLaw(grid; rate_factor = A³ᴰ)

        η², η³ = CenterField(σgrid), CenterField(σgrid)
        compute_viscosity!(η², two_dimensional, u, v, FreeSlipWalls())
        compute_viscosity!(η³, three_dimensional, u, v, FreeSlipWalls())
        @test interior_array(η²) == interior_array(η³)

        compute_viscosity!(η², two_dimensional, u, v, βᵘ, βᵛ, uᵇ, vᵇ, FreeSlipWalls())
        compute_viscosity!(η³, three_dimensional, u, v, βᵘ, βᵛ, uᵇ, vᵇ, FreeSlipWalls())
        @test interior_array(η²) == interior_array(η³)
    end

    @testset "Depth-varying A reaches the column integrals" begin
        grid = RectilinearGrid(arch; size = (8, 8), halo = (4, 4), x = (0, 1e5), y = (0, 1e5), topology = (Periodic, Periodic, Flat))
        σgrid = SigmaGrid(grid; Nσ = 8)
        u, v = XFaceField(grid), YFaceField(grid)
        set!(u, (x, y) -> 1e-6 * sin(2π * x / 1e5))
        fill_halo_regions!((u, v))
        A = CenterField(σgrid)
        A₀ = 1e-24
        set!(A, (x, y, ζ) -> A₀ * (1 + 7ζ))

        η₀, η = CenterField(σgrid), CenterField(σgrid)
        compute_viscosity!(η₀, GlenFlowLaw(grid; rate_factor = A₀), u, v, FreeSlipWalls())
        compute_viscosity!(η, GlenFlowLaw(grid; rate_factor = A), u, v, FreeSlipWalls())

        H = CenterField(grid)
        set!(H, 1000)
        η̄H, F₂ = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
        compute_column_integrals!(η̄H, F₂, η, H)

        ζ = Array(znodes(σgrid, Center()))
        ζᶠ = Array(znodes(σgrid, Face()))
        Δζ = diff(ζᶠ)
        scale = (1 .+ 7 .* ζ) .^ (-1 / 3)
        η₀ᵢ = interior_array(η₀)[3, 3, 1]
        @test interior_array(η)[3, 3, :] ≈ η₀ᵢ .* scale rtol = 1e-13
        @test interior_array(η̄H)[3, 3, 1] ≈ 1000 * η₀ᵢ * sum(Δζ .* scale) rtol = 1e-13
        weights = ((1 .- ζᶠ[1:end-1]) .^ 3 .- (1 .- ζᶠ[2:end]) .^ 3) ./ 3
        @test interior_array(F₂)[3, 3, 1] ≈ 1000 * sum(weights ./ scale) / η₀ᵢ rtol = 1e-13
    end

    @testset "DIVA shear follows A(ζ)" begin
        coarse = depth_varying_slab_error(10)
        fine = depth_varying_slab_error(40)
        @info "DIVA slab with depth-varying A" coarse fine
        @test 0 < fine.iterations < 300
        @test fine.profile < 1e-2
        @test fine.mean < 1e-2
        @test fine.profile < coarse.profile / 4
    end

    @testset "Model diagnoses A from the enthalpy" begin
        no_thermodynamics_grid = RectilinearGrid(arch; size = (4, 4), halo = (4, 4), x = (0, 1e5), y = (0, 1e5), topology = (Bounded, Bounded, Flat))
        rheology = GlenFlowLaw(no_thermodynamics_grid; rate_factor = ArrheniusRateFactor(no_thermodynamics_grid))
        @test_throws ArgumentError IceSheetModel(no_thermodynamics_grid; rheology)

        # A new model has the rate factor of its surface enthalpy, so a solve before set! is finite.
        model = rate_factor_model()
        A = model.rheology.rate_factor.field
        @test all(interior_array(A) .≈ cold_branch(243.15))

        # The pressure-adjusted temperature, with the Clausius-Clapeyron slope on.
        model = rate_factor_model(clausius_clapeyron = 7.9e-8)
        set!(model; H = 1000, T = 253.15)
        A = model.rheology.rate_factor.field
        ρg = model.constants.ice_density * model.constants.gravitational_acceleration
        ζ = Array(znodes(model.σgrid, Center()))
        expected = cold_branch.(253.15 .+ 7.9e-8 .* ρg .* 1000 .* (1 .- ζ))
        @test interior_array(A)[5, 2, :] ≈ expected rtol = 1e-12

        # Temperate ice: T′ = T₀ and the water softening, up to the cap.
        p = ρg .* 1000 .* (1 .- ζ)
        parameters = model.thermodynamics.parameters
        set!(model; E = (x, y, z) -> pressure_melting_enthalpy(ρg * 1000 * (1 - z), parameters) + 0.005 * parameters.latent_heat)
        @test interior_array(A)[5, 2, :] ≈ fill(warm_branch(273.15) * (1 + 181.25 * 0.005), length(p)) rtol = 1e-9

        set!(model; E = (x, y, z) -> pressure_melting_enthalpy(ρg * 1000 * (1 - z), parameters) + 0.05 * parameters.latent_heat)
        @test interior_array(A)[5, 2, :] ≈ fill(warm_branch(273.15) * (1 + 181.25 * 0.01), length(p)) rtol = 1e-9
    end

    @testset "Stress balance feels the temperature" begin
        for stress_balance in (DepthIntegrated(), DepthIntegrated(shear = DIVAShear()))
            arrhenius = rate_factor_model(; stress_balance)
            set!(arrhenius; H = sloped_thickness, T = 253.15)

            prescribed = rate_factor_model(; stress_balance, rate_factor = grid -> cold_branch(253.15))
            set!(prescribed; H = sloped_thickness, T = 253.15)
            @test interior_array(arrhenius.velocities.u) ≈ interior_array(prescribed.velocities.u) rtol = 1e-9

            warm = rate_factor_model(; stress_balance)
            set!(warm; H = sloped_thickness, T = 268.15)
            @test maximum(abs, interior_array(warm.velocities.u)) > maximum(abs, interior_array(arrhenius.velocities.u))
        end

        # A warm base carries the DIVA shear: the speed rises faster near the bed than with a warm surface.
        warm_base = rate_factor_model(stress_balance = DepthIntegrated(shear = DIVAShear()))
        warm_surface = rate_factor_model(stress_balance = DepthIntegrated(shear = DIVAShear()))
        set!(warm_base; H = sloped_thickness, T = (x, y, ζ) -> 268.15 - 30ζ)
        set!(warm_surface; H = sloped_thickness, T = (x, y, ζ) -> 238.15 + 30ζ)
        @test maximum(abs, interior_array(warm_base.velocities.u)) > maximum(abs, interior_array(warm_surface.velocities.u))
    end

    @testset "Energy steps update A" begin
        model = rate_factor_model(stress_balance = DepthIntegrated(shear = DIVAShear()), surface_temperature = 263.15, geothermal_flux = 0.1)
        set!(model; H = sloped_thickness, T = 263.15)
        A = model.rheology.rate_factor.field
        A⁰ = interior_array(A)
        reference = rate_factor_model(stress_balance = DepthIntegrated(shear = DIVAShear()), surface_temperature = 263.15, geothermal_flux = 0.1,
                                      rate_factor = grid -> arrhenius_rate_factor(263.15, 0.0, ArrheniusRateFactor(grid).parameters))
        set!(reference; H = sloped_thickness, T = 263.15)
        # H diffuses at about ρ g H² / C = 3.6 m² s⁻¹, so Δt must stay well below Δx² / D ≈ 3e7 s.
        for _ in 1:20
            time_step!(model, 1e7)
            time_step!(reference, 1e7)
        end
        T = interior_array(model.thermodynamics.state.T)
        A⁺ = interior_array(A)
        parameters = model.rheology.rate_factor.parameters
        @test all(isfinite, A⁺)
        @test A⁺ ≈ arrhenius_rate_factor.(T, interior_array(model.thermodynamics.state.ω), Ref(parameters)) rtol = 1e-13
        @test minimum(A⁺ ./ A⁰) ≥ 1 - 1e-12
        @test maximum(A⁺ ./ A⁰) > 1.01
        @test maximum(abs, interior_array(model.velocities.u)) > maximum(abs, interior_array(reference.velocities.u))
    end

    @testset "Restored state has the rate factor of the restored enthalpy" begin
        model = rate_factor_model()
        set!(model; H = sloped_thickness, T = (x, y, ζ) -> 268.15 - 30ζ)
        time_step!(model, 1e6)
        saved = Oceananigans.prognostic_state(model)

        restored = rate_factor_model()
        Oceananigans.restore_prognostic_state!(restored, saved)
        @test interior_array(restored.rheology.rate_factor.field) == interior_array(model.rheology.rate_factor.field)
        @test interior_array(restored.thermodynamics.state.T) == interior_array(model.thermodynamics.state.T)
    end
end
