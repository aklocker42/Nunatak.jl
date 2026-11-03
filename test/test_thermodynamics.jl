using Nunatak: enthalpy, temperature, water_fraction, pressure_melting_enthalpy, pressure_melting_temperature

function column_model(; H₀ = 1000, Nσ = 10, bed_refinement = 1, geothermal_flux = 0.042, surface_temperature = 243.15,
                      size = (1, 1), grounding = nothing, bed = 0, constants = nothing, schedule = IterationInterval(1),
                      surface_mass_balance = 0, grounded_melt_coupling = false, maximum_drainage_rate = 1.58443823077064e-9)
    grid = RectilinearGrid(arch; size, x = (0, 1e3), y = (0, 1e3), topology = (Periodic, Periodic, Flat))
    thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux, surface_temperature, schedule, grounded_melt_coupling,
                                            maximum_drainage_rate)
    model_constants = isnothing(constants) ? IceSheetConstants(grid; ice_density = 910) : constants
    model = IceSheetModel(grid; Nσ, bed_refinement, thermodynamics, grounding, bed, constants = model_constants,
                          velocity_solver = nothing, advection = UpwindBiased(order = 1), surface_mass_balance)
    set!(model; H = H₀)
    return model
end

column(field) = Array(interior(field, 1, 1, :))
surface_value(field) = Array(interior(field))[1, 1, 1]
layer_centres(model) = collect(znodes(model.σgrid, Center()))
layer_faces(model) = collect(znodes(model.σgrid, Face()))

# ρ ∫E dz per unit area of the first column.
function column_energy(model, H₀)
    Δζ = diff(layer_faces(model))
    return model.constants.ice_density * H₀ * sum(Δζ .* column(model.thermodynamics.state.E))
end

# Upward heat flux through the surface of a cold column, from the temperature of the top layer.
function surface_heat_flux(model, H₀)
    (; parameters) = model.thermodynamics
    ζᴺ = layer_centres(model)[end]
    Eᴺ = column(model.thermodynamics.state.E)[end]
    Tₛ = surface_value(model.thermodynamics.surface_temperature)
    Eₛ = parameters.heat_capacity * (Tₛ - parameters.reference_temperature)
    return parameters.thermal_conductivity / parameters.heat_capacity * (Eᴺ - Eₛ) / (H₀ * (1 - ζᴺ))
end

# Ice divide in the middle of three cells: uniform divergence u = γ (x - x₀) balanced by accumulation there. The middle
# column is upwind of its neighbours, so it stays exactly one dimensional. Velocities vanish at the walls.
function divide_model(; H₀, a₀, Nσ, geothermal_flux, surface_temperature)
    grid = RectilinearGrid(arch; size = (3, 1), x = (0, 3e3), y = (0, 1e3), topology = (Bounded, Periodic, Flat))
    γ = a₀ / H₀
    divide_velocity(x) = ifelse(0 < x < 3e3, γ * (x - 1500), zero(x))
    surface_mass_balance(x, y) = ifelse(abs(x - 1500) < 500, a₀, -a₀ / 2)
    thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux, surface_temperature)
    model = IceSheetModel(grid; Nσ, bed_refinement = 1, thermodynamics, constants = IceSheetConstants(grid; ice_density = 910),
                          velocity_solver = nothing, advection = UpwindBiased(order = 1), surface_mass_balance)
    set!(model.velocities.uσ, (x, y, ζ) -> divide_velocity(x))
    set!(model; H = H₀, u = (x, y) -> divide_velocity(x))
    return model
end

# Periodic or walled patch with variable thickness, flow that shears with depth, accumulation and melt.
function patch_model(topology; T = 250, surface_temperature = 250, geothermal_flux = 0)
    grid = RectilinearGrid(arch; size = (8, 8), halo = (4, 4), x = (0, 8e4), y = (0, 8e4), topology)
    L = 8e4
    ū(x, y) = 1e-6 * (1 + sin(2π * y / L) / 2)
    v̄(x, y) = 5e-7 * cos(2π * x / L)
    thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux, surface_temperature)
    model = IceSheetModel(grid; bed_refinement = 2, thermodynamics, constants = IceSheetConstants(grid; ice_density = 910),
                          velocity_solver = nothing, surface_mass_balance = (x, y) -> 1e-8 * (1 + cos(2π * x / L)), basal_melt_rate = 2e-9)
    set!(model.velocities.uσ, (x, y, ζ) -> ū(x, y) * (0.5 + ζ))
    set!(model.velocities.vσ, (x, y, ζ) -> v̄(x, y) * (0.5 + ζ))
    set!(model; H = (x, y) -> 1000 + 200 * sin(2π * x / L) * cos(2π * y / L), u = ū, v = v̄, T)
    return model
end

# Mechanical power of the driving stress minus the frictional and strain heat, relative to the power.
function dissipation_mismatch(stress_balance; advection = UpwindBiased(order = 1))
    Nx = 16
    L = 3.2e5
    grid = RectilinearGrid(arch; size = (Nx, Nx), halo = (4, 4), x = (0, L), y = (0, L), topology = (Periodic, Periodic, Flat))
    rheology = GlenFlowLaw(grid; rate_factor = 1e-24, strain_rate_regularization = 1e-16)
    friction = WeertmanFriction(grid; coefficient = 1e10, exponent = 1)
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000),
                                          picard_iteration_control = Convergence(; reltol = 1e-11, maxiter = 500))
    model = IceSheetModel(grid; stress_balance, rheology, friction, velocity_solver, advection, thermodynamics = EnthalpyThermodynamics(grid),
                          constants = IceSheetConstants(grid; ice_density = 910))
    set!(model; H = (x, y) -> 1000 + 100 * sin(2π * x / L) * cos(2π * y / L))
    H⁻ = Array(interior(model.geometry.H))
    Δt = 1e6
    time_step!(model, Δt)

    (; Φ, qᵍ, W) = model.thermodynamics.state
    Δζ = diff(layer_faces(model))
    H = Array(interior(model.geometry.H))
    strain = sum(reshape(Δζ, 1, 1, :) .* Array(interior(Φ)); dims = 3)
    area = (L / Nx)^2
    heat = area * sum(Array(interior(qᵍ)) .+ H .* strain)
    power = sum(Array(interior(model.auxiliary.τᵘ)) .* Array(interior(model.velocities.u))) +
            sum(Array(interior(model.auxiliary.τᵛ)) .* Array(interior(model.velocities.v)))
    surface_flux_error = maximum(abs, Array(interior(W))[:, :, end])
    thinning_rate = maximum(abs, H .- H⁻) / Δt
    return (; mismatch = abs(heat - power) / power, power, heat, surface_flux_error, thinning_rate)
end

simpson(f, a, b; N = 4000) = (b - a) / 3N * (f(a) + f(b) + sum((isodd(n) ? 4 : 2) * f(a + n * (b - a) / N) for n in 1:N-1))

@testset "Thermodynamics" begin
    @testset "Enthalpy conversions round trip" begin
        model = column_model()
        p = model.thermodynamics.parameters
        for (T, pressure) in ((243.15, 0.0), (263.15, 5e6), (270.0, 1e7))
            E = enthalpy(T, 0, pressure, p)
            @test temperature(E, pressure, p) ≈ T
            @test water_fraction(E, pressure, p) == 0
        end
        Pa = 1e7
        Tₚₘ = pressure_melting_temperature(Pa, p)
        @test Tₚₘ ≈ 273.15 - 7.9e-8 * Pa
        E = enthalpy(Tₚₘ, 0.01, Pa, p)
        @test water_fraction(E, Pa, p) ≈ 0.01
        @test temperature(E, Pa, p) ≈ Tₚₘ
        @test enthalpy(300.0, 0, Pa, p) ≈ pressure_melting_enthalpy(Pa, p)
    end

    @testset "Steady conduction is linear in height, on uniform and refined layers" begin
        for bed_refinement in (1, 2)
            H₀, G, Tₛ = 1000, 0.042, 243.15
            model = column_model(; H₀, geothermal_flux = G, surface_temperature = Tₛ, bed_refinement)
            for _ in 1:4
                time_step!(model, 1e14)
            end
            k = model.thermodynamics.parameters.thermal_conductivity
            exact = Tₛ .+ G .* H₀ .* (1 .- layer_centres(model)) ./ k
            @test maximum(abs, column(model.thermodynamics.state.T) .- exact) < 1e-8
            @test surface_value(model.thermodynamics.state.Hw) == 0
        end
    end

    @testset "Surface warming converges at second order to the Fourier series" begin
        H₀, G, Tₛ, ΔT = 1000, 0.042, 243.15, 10.0
        t_end = 6.3e10
        errors = map((10, 20, 40)) do Nσ
            model = column_model(; H₀, Nσ, geothermal_flux = G, surface_temperature = Tₛ)
            (; heat_capacity, thermal_conductivity) = model.thermodynamics.parameters
            k = thermal_conductivity
            set!(model; T = (x, y, ζ) -> Tₛ + G * H₀ * (1 - ζ) / k)
            set!(model.thermodynamics.surface_temperature, Tₛ + ΔT)
            Nt = 4 * Nσ^2
            for _ in 1:Nt
                time_step!(model, t_end / Nt)
            end
            κ = k / (model.constants.ice_density * heat_capacity)
            z = layer_centres(model) .* H₀
            steady = Tₛ + ΔT .+ G .* (H₀ .- z) ./ k
            transient = zero(z)
            for n in 0:200
                λ = (2n + 1) * π / (2H₀)
                transient .+= -ΔT * 4 / π * (-1)^n / (2n + 1) .* cos.(λ .* z) .* exp(-κ * λ^2 * t_end)
            end
            return maximum(abs, column(model.thermodynamics.state.T) .- (steady .+ transient))
        end
        @test errors[3] < 0.01
        @test errors[1] / errors[2] > 3
        @test errors[2] / errors[3] > 3
    end

    @testset "Energy budget closes with geothermal and frictional heating" begin
        H₀, G = 800, 0.05
        model = column_model(; H₀, geothermal_flux = G, Nσ = 12, bed_refinement = 2)
        set!(model; T = (x, y, ζ) -> 250 - 5ζ)
        β, ub = 1e8, 1e-6
        set!(model.auxiliary.βᵘ, β)
        set!(model.velocities.uᵇ, ub)
        F = β * ub^2
        Δt = 1e9
        for _ in 1:50
            energy⁻ = column_energy(model, H₀)
            time_step!(model, Δt)
            budget = column_energy(model, H₀) - energy⁻ - Δt * (G + F - surface_heat_flux(model, H₀))
            @test abs(budget) < 1e-11 * abs(energy⁻)
        end
    end

    @testset "Warm base melts at the analytic rate, then refreezes the stored water" begin
        H₀, G, Tₛ = 1000, 0.1, 263.15
        model = column_model(; H₀, geothermal_flux = G, surface_temperature = Tₛ, Nσ = 20)
        (; state, parameters) = model.thermodynamics
        (; thermal_conductivity, latent_heat) = parameters
        ρ = model.constants.ice_density
        Tₚₘ = pressure_melting_temperature(ρ * model.constants.gravitational_acceleration * H₀, parameters)

        for _ in 1:20
            time_step!(model, 1e13)
        end

        exact = (G - thermal_conductivity * (Tₚₘ - Tₛ) / H₀) / (ρ * latent_heat)
        @test exact > 0
        @test surface_value(state.mᵍ) ≈ exact rtol = 1e-3
        @test surface_value(state.Hw) > 0

        # Too little water for one step, so the column must pay back the excess latent heat itself.
        set!(state.Hw, 1e-2)
        set!(model.thermodynamics.geothermal_flux, 0)
        Δt = 1e9
        total_energy(model) = column_energy(model, H₀) + ρ * latent_heat * surface_value(state.Hw)
        froze_out = false
        for _ in 1:200
            energy⁻ = total_energy(model)
            time_step!(model, Δt)
            q = thermal_conductivity / parameters.heat_capacity *
                (column(state.E)[end] - parameters.heat_capacity * (Tₛ - parameters.reference_temperature)) / (H₀ * (1 - layer_centres(model)[end]))
            @test abs(total_energy(model) - energy⁻ + Δt * q) < 1e-10 * abs(energy⁻)
            froze_out |= surface_value(state.Hw) == 0
        end
        @test froze_out
        @test surface_value(state.mᵍ) == 0
    end

    @testset "Floating base sits at the ocean temperature with no grounded melt" begin
        H₀, Tₛ = 500, 253.15
        model = column_model(; H₀, surface_temperature = Tₛ, geothermal_flux = 0.05, Nσ = 10,
                             grounding = SubgridGroundedFraction(), bed = -3000)
        @test surface_value(model.geometry.φ) == 0
        for _ in 1:4
            time_step!(model, 1e14)
        end
        Tₒ = surface_value(model.thermodynamics.ocean_temperature)
        exact = Tₒ .+ (Tₛ - Tₒ) .* layer_centres(model)
        @test maximum(abs, column(model.thermodynamics.state.T) .- exact) < 1e-8
        @test surface_value(model.thermodynamics.state.mᵍ) == 0
    end

    @testset "Ice-free columns hold the surface enthalpy" begin
        Tₛ = 253.15
        model = column_model(; H₀ = 0.5, surface_temperature = Tₛ)
        set!(model; T = 240)
        time_step!(model, 1e9)
        @test all(column(model.thermodynamics.state.T) .≈ Tₛ)
        @test surface_value(model.thermodynamics.state.Hw) == 0
    end

    @testset "A sparse schedule steps over the elapsed time" begin
        Δt = 1e9
        every_step = column_model()
        sparse = column_model(; schedule = IterationInterval(2))
        for model in (every_step, sparse)
            set!(model; T = 250)
        end
        time_step!(every_step, 2Δt)
        # The schedule actuates at iterations 0 and 2, so the second energy step covers the two smaller ice steps.
        time_step!(sparse, 2Δt)
        time_step!(sparse, Δt)
        @test column(sparse.thermodynamics.state.E) == column(every_step.thermodynamics.state.E)
        time_step!(sparse, Δt)
        time_step!(every_step, 2Δt)
        @test column(sparse.thermodynamics.state.E) == column(every_step.thermodynamics.state.E)
    end

    @testset "update_state! outside a time step does not step the energy" begin
        model = column_model()
        time_step!(model, 1e9)
        E = column(model.thermodynamics.state.E)
        Oceananigans.TimeSteppers.update_state!(model)
        @test column(model.thermodynamics.state.E) == E
    end

    @testset "Prognostic state restores enthalpy and water" begin
        model = column_model()
        set!(model.thermodynamics.state.Hw, 2.5)
        time_step!(model, 1e9)
        saved = Oceananigans.prognostic_state(model)
        restored = column_model()
        Oceananigans.restore_prognostic_state!(restored, saved)
        @test column(restored.thermodynamics.state.E) == column(model.thermodynamics.state.E)
        @test surface_value(restored.thermodynamics.state.Hw) == surface_value(model.thermodynamics.state.Hw)
        @test surface_value(restored.thermodynamics.state.Hⁿ) == surface_value(model.thermodynamics.state.Hⁿ)
        @test restored.thermodynamics.last_step_time[] == model.thermodynamics.last_step_time[]
    end

    @testset "Accumulation over an ice divide converges at second order to the Robin solution" begin
        H₀, a₀, G, Tₛ = 1000, 1e-8, 0.042, 243.15
        errors = map((10, 20, 40)) do Nσ
            model = divide_model(; H₀, a₀, Nσ, geothermal_flux = G, surface_temperature = Tₛ)
            for _ in 1:1000
                time_step!(model, 1e10)
            end
            (; state, parameters) = model.thermodynamics
            @test Array(interior(state.W))[2, 1, end] ≈ -a₀ rtol = 1e-8
            @test Array(interior(model.geometry.H))[2, 1, 1] ≈ H₀
            k = parameters.thermal_conductivity
            κ = k / (model.constants.ice_density * parameters.heat_capacity)
            l² = 2H₀ * κ / a₀
            exact = [Tₛ + G / k * simpson(z′ -> exp(-z′^2 / l²), ζ * H₀, H₀) for ζ in layer_centres(model)]
            return maximum(abs, Array(interior(state.T, 2, 1, :)) .- exact)
        end
        @info "Robin column maximum error (K), Nσ = 10, 20, 40" errors
        @test errors[3] < 0.01
        @test errors[1] / errors[2] > 3
        @test errors[2] / errors[3] > 3
    end

    @testset "A thickening or thinning column follows the heat equation in fixed height" begin
        H₀, G, T₀, B = 1000, 0.04, 240, 5
        λ = π / 1000
        t_end = 2e10
        for a₀ in (1e-8, -1e-8)
            errors = map((10, 20, 40)) do Nσ
                model = column_model(; H₀, Nσ, geothermal_flux = G, surface_mass_balance = a₀)
                (; parameters, state, surface_temperature) = model.thermodynamics
                k = parameters.thermal_conductivity
                κ = k / (model.constants.ice_density * parameters.heat_capacity)
                exact(z, t) = T₀ - G * z / k + B * exp(-κ * λ^2 * t) * cos(λ * z)
                set!(model; T = (x, y, ζ) -> exact(ζ * H₀, 0))
                Nt = 4 * Nσ^2
                Δt = t_end / Nt
                for n in 1:Nt
                    set!(surface_temperature, exact(H₀ + a₀ * n * Δt, n * Δt))
                    time_step!(model, Δt)
                end
                H = surface_value(model.geometry.H)
                @test H ≈ H₀ + a₀ * t_end
                return maximum(abs, column(state.T) .- exact.(layer_centres(model) .* H, t_end))
            end
            @info "Moving-surface column maximum error (K), Nσ = 10, 20, 40" a₀ errors
            @test errors[3] < 0.03
            @test errors[1] / errors[2] > 2.5
            @test errors[2] / errors[3] > 2.5
        end
    end

    @testset "Uniform enthalpy stays uniform under variable thickness, shear, accumulation and melt" begin
        model = patch_model((Periodic, Periodic, Flat))
        E₀ = Array(interior(model.thermodynamics.state.E))[1, 1, 1]
        H⁻ = Array(interior(model.geometry.H))
        for _ in 1:20
            time_step!(model, 1e9)
        end
        @test maximum(abs, Array(interior(model.geometry.H)) .- H⁻) > 10
        @test maximum(abs, Array(interior(model.thermodynamics.state.E)) .- E₀) < 1e-9 * E₀
    end

    @testset "Global energy budget closes through the boundary and advective fluxes" begin
        for topology in ((Periodic, Periodic, Flat), (Bounded, Bounded, Flat))
            G, Tₛ, m = 0.05, 245, 2e-9
            model = patch_model(topology; geothermal_flux = G, surface_temperature = Tₛ, T = (x, y, ζ) -> 240 + 15ζ + 5 * sin(2π * x / 8e4))
            (; state, parameters) = model.thermodynamics
            ρ = model.constants.ice_density
            ζᶠ = layer_faces(model)
            Δζ = reshape(diff(ζᶠ), 1, 1, :)
            ζᴺ = layer_centres(model)[end]
            Kᶜ = parameters.thermal_conductivity / parameters.heat_capacity
            Eₛ = parameters.heat_capacity * (Tₛ - parameters.reference_temperature)
            energy() = ρ * sum(Array(interior(model.geometry.H)) .* Δζ .* Array(interior(state.E)))
            Δt = 1e9
            for _ in 1:10
                energy⁻ = energy()
                time_step!(model, Δt)
                E = Array(interior(state.E))
                H = Array(interior(model.geometry.H))[:, :, 1]
                Wˢ = Array(interior(state.W))[:, :, end]
                conduction = Kᶜ .* (E[:, :, end] .- Eₛ) ./ (H .* (1 - ζᴺ))
                advection = ρ .* (max.(Wˢ, 0) .* E[:, :, end] .- max.(-Wˢ, 0) .* Eₛ)
                expected = Δt * sum(G .- conduction .- advection .- ρ * m .* E[:, :, 1])
                @test abs(energy() - energy⁻ - expected) < 1e-11 * energy⁻
            end
        end
    end

    @testset "Strong accumulation over a sharp front creates no new extrema" begin
        H₀ = 1000
        model = column_model(; H₀, Nσ = 10, geothermal_flux = 0, surface_temperature = 250, surface_mass_balance = 2e-7)
        set!(model; T = (x, y, ζ) -> ifelse(ζ < 0.5, 230, 260))
        for _ in 1:20
            time_step!(model, 1e8)
            T = column(model.thermodynamics.state.T)
            @test minimum(T) ≥ 230 - 1e-8
            @test maximum(T) ≤ 260 + 1e-8
        end
    end

    @testset "Upwind transport at unit Courant number shifts each layer by one cell" begin
        Nx, Δx = 8, 1e3
        for (name, size, position, shift) in ((:x, (Nx, 1), (x, y) -> x, (1, 0)), (:y, (1, Nx), (x, y) -> y, (0, 1)))
            grid = RectilinearGrid(arch; size, x = (0, size[1] * Δx), y = (0, size[2] * Δx), topology = (Periodic, Periodic, Flat))
            thermodynamics = EnthalpyThermodynamics(grid; thermal_conductivity = 1e-12)
            model = IceSheetModel(grid; Nσ = 4, bed_refinement = 1, thermodynamics, velocity_solver = nothing, advection = UpwindBiased(order = 1))
            Δt = 1e9
            layer_speed(ζ) = ifelse(ζ > 0.5, Δx / Δt, zero(ζ))
            field = name == :x ? model.velocities.uσ : model.velocities.vσ
            set!(field, (x, y, ζ) -> layer_speed(ζ))
            set!(model; H = 500, T = (x, y, ζ) -> 250 + 10 * sin(2π * position(x, y) / (Nx * Δx)))
            E⁻ = Array(interior(model.thermodynamics.state.E))
            time_step!(model, Δt)
            E = Array(interior(model.thermodynamics.state.E))
            @test E[:, :, 1:2] ≈ E⁻[:, :, 1:2] rtol = 1e-10
            @test E[:, :, 3:4] ≈ circshift(E⁻[:, :, 3:4], (shift..., 0)) rtol = 1e-10
            @test maximum(abs, Array(interior(model.thermodynamics.state.W))) < 1e-18
        end
    end

    @testset "Friction and strain heat account for the work of the driving stress" begin
        for stress_balance in (DepthIntegrated(), DepthIntegrated(shear = DIVAShear()))
            result = dissipation_mismatch(stress_balance)
            @info "Dissipation against driving power" summary(stress_balance) result
            @test result.mismatch < 1e-8
        end
    end

    @testset "A temperate layer above the base insulates it, so all basal heat melts" begin
        H₀, G = 1000, 0.08
        model = column_model(; H₀, geothermal_flux = G, surface_temperature = 263.15, Nσ = 20, maximum_drainage_rate = 0)
        (; state, parameters) = model.thermodynamics
        ρ = model.constants.ice_density
        set!(state.Hw, 1)
        c, L, Tᵣ, T₀ = parameters.heat_capacity, parameters.latent_heat, parameters.reference_temperature, parameters.melting_temperature
        set!(model; E = (x, y, ζ) -> ifelse(ζ < 0.2, c * (T₀ - Tᵣ) + L / 50, c * (260 - Tᵣ)))
        @test surface_value(state.ω) > 0
        time_step!(model, 1e8)
        @test surface_value(state.sᵇ) == 1
        @test surface_value(state.mᵍ) ≈ G / (ρ * parameters.latent_heat) rtol = 1e-12
    end

    @testset "Temperate layers drain towards the drainage water fraction into the basal water" begin
        H₀ = 1000
        model = column_model(; H₀, geothermal_flux = 0, surface_temperature = 263.15, Nσ = 10)
        (; state, parameters) = model.thermodynamics
        ω_d, Rmax = parameters.drainage_water_fraction, parameters.maximum_drainage_rate
        @test Nunatak.drainage_rate(ω_d / 2, parameters) == 0
        @test Nunatak.drainage_rate(2ω_d, parameters) ≈ Rmax / 10
        @test Nunatak.drainage_rate(5ω_d, parameters) ≈ Rmax

        c, L, Tᵣ, T₀ = parameters.heat_capacity, parameters.latent_heat, parameters.reference_temperature, parameters.melting_temperature
        set!(model; E = c * (T₀ - Tᵣ) + 5ω_d * L)
        @test minimum(column(state.ω)) > 3ω_d
        Δt = 1
        time_step!(model, Δt)
        @test surface_value(state.Hw) ≈ H₀ * Rmax * Δt rtol = 1e-8
        for _ in 1:100
            time_step!(model, 1e9)
        end
        @test maximum(column(state.ω)) ≤ ω_d + 1e-12
        @test surface_value(state.Hw) > 0.1 * H₀ * ω_d
    end

    @testset "Coupled grounded melt moves mass between the ice and the basal water" begin
        H₀, G, Tₛ = 1000, 0.1, 263.15
        coupled = column_model(; H₀, geothermal_flux = G, surface_temperature = Tₛ, Nσ = 20, grounded_melt_coupling = true)
        fixed = column_model(; H₀, geothermal_flux = G, surface_temperature = Tₛ, Nσ = 20)
        for model in (coupled, fixed)
            set!(model; T = (x, y, ζ) -> Tₛ + 10 * (1 - ζ))
        end
        Δt = 1e10
        for _ in 1:30
            for model in (coupled, fixed)
                time_step!(model, Δt)
            end
            (; Hw, mᵍ) = coupled.thermodynamics.state
            @test surface_value(coupled.geometry.H) + surface_value(Hw) - Δt * surface_value(mᵍ) ≈ H₀ rtol = 1e-12
            @test surface_value(fixed.geometry.H) == H₀
        end
        @test surface_value(coupled.thermodynamics.state.mᵍ) > 0
        @test surface_value(coupled.geometry.H) < H₀
    end

    @testset "A cold base leaves the thickness as a model without thermodynamics" begin
        grid = RectilinearGrid(arch; size = (8, 1), x = (0, 8e3), y = (0, 1e3), topology = (Bounded, Periodic, Flat))
        common = (; velocity_solver = nothing, advection = UpwindBiased(order = 1))
        plain = IceSheetModel(grid; common...)
        thermal = IceSheetModel(grid; thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux = 0.05), common...)
        @test plain.thermodynamics === nothing
        for model in (plain, thermal)
            set!(model; H = (x, y) -> 100 + x / 100, u = 1e-6)
            for _ in 1:5
                time_step!(model, 1e7)
            end
        end
        @test Array(interior(plain.geometry.H)) == Array(interior(thermal.geometry.H))
    end
end
