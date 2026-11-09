maybe_step_thermodynamics!(model, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Advance the energy once per ice time step, in the final stage of `update_state!` when the velocities are current.
The time step is the time elapsed since the last energy step, so it stays correct for a sparse `schedule`.
"""
function maybe_step_thermodynamics!(model, thermodynamics::EnthalpyThermodynamics)
    final_stage = model.clock.stage == model.timestepper.Nstages
    Δt = model.clock.time - thermodynamics.last_step_time[]
    (final_stage && Δt > 0 && thermodynamics.schedule(model)) || return nothing
    step_thermodynamics!(model, thermodynamics, convert(eltype(model.grid), Δt))
    thermodynamics.last_step_time[] = model.clock.time
    return nothing
end

basal_drag(::SSA, auxiliary) = (auxiliary.βᵘ, auxiliary.βᵛ)
basal_drag(::DIVA, auxiliary) = (auxiliary.βᵇᵘ, auxiliary.βᵇᵛ)
basal_drag(::ShallowIce, auxiliary) = (auxiliary.βᵘ, auxiliary.βᵛ)

"""
$(TYPEDSIGNATURES)

Compute the strain heating from the current velocities, then advance the enthalpy by `Δt`.
"""
function step_thermodynamics!(model, thermodynamics, Δt)
    (; uᵇ, vᵇ, uσ, vσ) = model.velocities
    fill_halo_regions!((basal_drag(model.stress_balance, model.auxiliary)..., uᵇ, vᵇ))
    fill_halo_regions!((uσ, vσ))
    compute_strain_heating!(thermodynamics.state.Φ, model, model.stress_balance)
    advance_enthalpy!(model, thermodynamics, Δt)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Advance the enthalpy by `Δt` with the strain heating `Φ` already in the state, with one implicit tridiagonal solve
per column, using finite volumes on the layers:

`ρ (H⁺ E⁺ - Hⁿ Eⁿ) Δζ / Δt + ρ Δζ ∇⋅(Hⁿ u Eⁿ) + Q[k+1] - Q[k] = H⁺ Δζ Φ`.

Horizontal transport is explicit first-order upwind. `Q` holds vertical advection with the volume flux `W` from
`compute_layer_volume_fluxes!`, which makes uniform `E` an exact solution, and conduction `-K ∂E/∂z` with the
diffusivity `K` lagged. Harmonic face averaging keeps the flux consistent across the cold-temperate transition.
The basal state then gives the melt rate, temperate layers drain above `drainage_water_fraction`, and the melt and
drainage fill the water layer.
"""
function advance_enthalpy!(model, thermodynamics, Δt)
    (; grid, σgrid, constants) = model
    (; parameters, geothermal_flux, surface_temperature, ocean_temperature) = thermodynamics
    (; E, Hw, mᵍ, rᵇ, sᵇ, Dᵇ, qᵍ, Hⁿ, W, Φ, right_hand_side, column_solver) = thermodynamics.state
    (; H) = model.geometry
    (; uᵇ, vᵇ, uσ, vσ) = model.velocities
    φ = floating_melt_fraction(model.grounding, model.geometry)
    βᵘ, βᵛ = basal_drag(model.stress_balance, model.auxiliary)
    arch = architecture(grid)

    fill_halo_regions!(E)

    launch!(arch, grid, :xy, _compute_layer_volume_fluxes!, W, grid, σgrid, H, Hⁿ, uσ, vσ, model.basal_melt_rate, φ,
            coupled_grounded_melt(thermodynamics), parameters, Δt)
    launch!(arch, grid, :xy, _compute_basal_state!, rᵇ, sᵇ, Dᵇ, qᵍ, σgrid, E, H, Hw, φ, geothermal_flux, βᵘ, βᵛ, uᵇ, vᵇ, parameters, constants)

    launch!(arch, σgrid, :xyz, _compute_column_system!, column_solver.a, column_solver.b, column_solver.c, right_hand_side,
            grid, σgrid, E, H, Hⁿ, W, Φ, uσ, vσ, φ, rᵇ, sᵇ, Dᵇ, qᵍ, surface_temperature, ocean_temperature, parameters, constants, Δt)

    Oceananigans.Solvers.solve!(E, column_solver, right_hand_side)

    launch!(arch, grid, :xy, _update_basal_water!, Hw, mᵍ, E, σgrid, H, φ, rᵇ, sᵇ, Dᵇ, qᵍ, parameters, constants, Δt)

    update_thermal_diagnostics!(model, thermodynamics)
    parent(Hⁿ) .= parent(H)
    return nothing
end

update_thermal_diagnostics!(model, ::Nothing) = nothing

"""
$(TYPEDSIGNATURES)

Diagnose the temperature and water fraction from the enthalpy, then the rate factor of the rheology if it follows them.
"""
function update_thermal_diagnostics!(model, thermodynamics::EnthalpyThermodynamics)
    (; σgrid, constants) = model
    (; H) = model.geometry
    (; parameters) = thermodynamics
    (; E, T, ω) = thermodynamics.state
    launch!(architecture(σgrid), σgrid, :xyz, _compute_thermal_diagnostics!, T, ω, E, σgrid, H, parameters, constants)
    update_rate_factor!(model.rheology, T, ω, σgrid, H, parameters, constants)
    return nothing
end

update_rate_factor!(rheology, T, ω, σgrid, H, parameters, constants) = nothing
update_rate_factor!(rheology::GlenFlowLaw, args...) = update_rate_factor!(rheology.rate_factor, args...)

function update_rate_factor!(rate_factor::ArrheniusRateFactor, T, ω, σgrid, H, parameters, constants)
    launch!(architecture(σgrid), σgrid, :xyz, _compute_rate_factor!, rate_factor.field, rate_factor.parameters, T, ω, σgrid, H, parameters, constants)
    fill_halo_regions!(rate_factor.field)
    return nothing
end

# T′ = T - Tₚₘ + T₀, the temperature relative to the melting point at this pressure.
@kernel function _compute_rate_factor!(A, arrhenius, T, ω, σgrid, H, parameters, constants)
    i, j, k = @index(Global, NTuple)
    p = layer_pressure(i, j, k, σgrid, H, constants)
    T′ = @inbounds T[i, j, k] - pressure_melting_temperature(p, parameters) + parameters.melting_temperature
    @inbounds A[i, j, k] = arrhenius_rate_factor(T′, ω[i, j, k], arrhenius)
end

@kernel function _compute_thermal_diagnostics!(T, ω, E, σgrid, H, parameters, constants)
    i, j, k = @index(Global, NTuple)
    p = layer_pressure(i, j, k, σgrid, H, constants)
    @inbounds T[i, j, k] = temperature(E[i, j, k], p, parameters)
    @inbounds ω[i, j, k] = water_fraction(E[i, j, k], p, parameters)
end

@inline grounded_part(i, j, ::Nothing) = 1
@inline grounded_part(i, j, φ) = @inbounds φ[i, j, 1]

# Cell mean of τᵇ⋅uᵇ = β uᵇ² from the velocity points.
@inline function frictional_heating(i, j, βᵘ, βᵛ, uᵇ, vᵇ)
    return @inbounds (βᵘ[i, j, 1] * uᵇ[i, j, 1]^2 + βᵘ[i+1, j, 1] * uᵇ[i+1, j, 1]^2 +
                      βᵛ[i, j, 1] * vᵇ[i, j, 1]^2 + βᵛ[i, j+1, 1] * vᵇ[i, j+1, 1]^2) / 2
end

# Base held at Eₚₘ when r = 1, and receiving the heat flux qᵍ when r = 0. Wet bases stay at Eₚₘ until the water has refrozen.
# s is the temperate fraction of layer 1, over which a melting base becomes insulating.
@kernel function _compute_basal_state!(rᵇ, sᵇ, Dᵇ, qᵍ, σgrid, E, H, Hw, φ, G, βᵘ, βᵛ, uᵇ, vᵇ, parameters, constants)
    i, j = @index(Global, NTuple)
    @inbounds begin
        H̃ = column_thickness(i, j, H, parameters)
        ζ₁ = znode(1, 1, 1, σgrid, Center(), Center(), Center())
        D = layer_diffusivity(i, j, 1, σgrid, E, H, parameters, constants) / (H̃ * ζ₁)
        F = frictional_heating(i, j, βᵘ, βᵛ, uᵇ, vᵇ)
        Eₚₘ = pressure_melting_enthalpy(basal_pressure(i, j, H, constants), parameters)
        Eᵇ = E[i, j, 1] + (G[i, j, 1] + F) / D
        r = max(unit_ramp(Eᵇ - Eₚₘ, parameters.enthalpy_blend_width), unit_ramp(Hw[i, j, 1], parameters.water_layer_blend_thickness))
        E₁ₚₘ = pressure_melting_enthalpy(layer_pressure(i, j, 1, σgrid, H, constants), parameters)
        δE = parameters.enthalpy_blend_width
        Dᵇ[i, j, 1] = D
        rᵇ[i, j, 1] = r
        sᵇ[i, j, 1] = unit_ramp(E[i, j, 1] - E₁ₚₘ + δE, δE)
        qᵍ[i, j, 1] = grounded_part(i, j, φ) * G[i, j, 1] + F
    end
end

# Conductance of the interior face k between layers k-1 and k: series resistance of the two half layers.
@inline function face_conductance(i, j, k, σgrid, E, H, parameters, constants)
    H̃ = column_thickness(i, j, H, parameters)
    ζᶠ = znode(1, 1, k, σgrid, Center(), Center(), Face())
    ζ⁻ = znode(1, 1, k-1, σgrid, Center(), Center(), Center())
    ζ⁺ = znode(1, 1, k, σgrid, Center(), Center(), Center())
    K⁻ = layer_diffusivity(i, j, k-1, σgrid, E, H, parameters, constants)
    K⁺ = layer_diffusivity(i, j, k, σgrid, E, H, parameters, constants)
    return 1 / (H̃ * ((ζᶠ - ζ⁻) / K⁻ + (ζ⁺ - ζᶠ) / K⁺))
end

@inline function surface_conductance(i, j, σgrid, E, H, parameters, constants)
    H̃ = column_thickness(i, j, H, parameters)
    Nσ = size(σgrid, 3)
    ζ = znode(1, 1, Nσ, σgrid, Center(), Center(), Center())
    return layer_diffusivity(i, j, Nσ, σgrid, E, H, parameters, constants) / (H̃ * (1 - ζ))
end

# Off-diagonal couplings at interior face k: the entry for E[k-1] in row k and for E[k] in row k-1. Advection is
# centered, with just enough added diffusion to keep both non-positive, so the solve cannot create new extrema.
@inline function face_couplings(i, j, k, σgrid, E, H, W, parameters, constants)
    ζᶠ = znode(1, 1, k, σgrid, Center(), Center(), Face())
    ζ⁻ = znode(1, 1, k-1, σgrid, Center(), Center(), Center())
    ζ⁺ = znode(1, 1, k, σgrid, Center(), Center(), Center())
    θ = (ζᶠ - ζ⁻) / (ζ⁺ - ζ⁻)
    F = @inbounds constants.ice_density * W[i, j, k]
    D = max(face_conductance(i, j, k, σgrid, E, H, parameters, constants), max(F, zero(F)) * θ, max(-F, zero(F)) * (1 - θ))
    return -F * (1 - θ) - D, F * θ - D
end

# The basal flux into layer 1 is S - Λ E[1] and the surface flux out of layer Nσ is Dˢ E[Nσ] - Sˢ. χ weights the
# grounded melting-point condition, which a temperate layer above the base switches off. Ice leaving through
# either boundary carries the enthalpy of its layer, ice arriving carries the boundary enthalpy.
@kernel function _compute_column_system!(a, b, c, f, grid, σgrid, E, H, Hⁿ, W, Φ, uσ, vσ, φ, rᵇ, sᵇ, Dᵇ, qᵍ, Tₛ, Tₒ, parameters, constants, Δt)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        Nσ = size(σgrid, 3)
        ρ = constants.ice_density
        Δζ = layer_thickness(k, σgrid)
        H̃⁺ = column_thickness(i, j, H, parameters)
        H̃ⁿ = column_thickness(i, j, Hⁿ, parameters)
        μ = column_has_ice(i, j, H, parameters)
        Eₛ = surface_enthalpy(i, j, Tₛ, parameters)

        ϕ = grounded_part(i, j, φ)
        r = rᵇ[i, j, 1]
        χ = ϕ * r * (1 - sᵇ[i, j, 1])
        Eₚₘ = pressure_melting_enthalpy(basal_pressure(i, j, H, constants), parameters)
        Eₒ = enthalpy(Tₒ[i, j, 1], zero(Eₛ), basal_pressure(i, j, H, constants), parameters)
        Fᵇ = ρ * W[i, j, 1]
        Λ = Dᵇ[i, j, 1] * (χ + 1 - ϕ) + max(-Fᵇ, zero(Fᵇ))
        S = (1 - r) * qᵍ[i, j, 1] + Dᵇ[i, j, 1] * (χ * Eₚₘ + (1 - ϕ) * Eₒ) + max(Fᵇ, zero(Fᵇ)) * Eₚₘ

        Fˢ = ρ * W[i, j, Nσ+1]
        Kˢ = surface_conductance(i, j, σgrid, E, H, parameters, constants)
        Dˢ = Kˢ + max(Fˢ, zero(Fˢ))
        Sˢ = (Kˢ + max(-Fˢ, zero(Fˢ))) * Eₛ

        _, upper⁻ = face_couplings(i, j, k, σgrid, E, H, W, parameters, constants)
        lower⁺, upper⁺ = face_couplings(i, j, k+1, σgrid, E, H, W, parameters, constants)

        diagonal⁻ = ifelse(k == 1, Λ, -upper⁻)
        diagonal⁺ = ifelse(k == Nσ, Dˢ, -lower⁺)
        forcing = ifelse(k == 1, S, zero(S)) + ifelse(k == Nσ, Sˢ, zero(S))
        transport = ρ * Δζ * enthalpy_flux_divergence(i, j, k, grid, uσ, vσ, Hⁿ, E)

        a[i, j, k] = ifelse(μ, lower⁺, zero(S))
        c[i, j, k] = ifelse(μ, upper⁺, zero(S))
        b[i, j, k] = ifelse(μ, ρ * H̃⁺ * Δζ / Δt + diagonal⁻ + diagonal⁺, one(S))
        f[i, j, k] = ifelse(μ, ρ * H̃ⁿ * Δζ * E[i, j, k] / Δt - transport + H̃⁺ * Δζ * Φ[i, j, k] + forcing, Eₛ)
    end
end

# Fraction of the largest drainage rate: zero up to ω_d, then linear ramps to 1/10 at 2ω_d and to 1 at 3ω_d,
# the piecewise-linear drainage function of Aschwanden et al. (2012).
@inline function drainage_rate(ω, parameters)
    x = ω / parameters.drainage_water_fraction
    shape = clamp(x - 1, zero(x), one(x)) / 10 + 9 * clamp(x - 2, zero(x), one(x)) / 10
    return parameters.maximum_drainage_rate * shape
end

# Melt from the heat left at the base, ρ L m = r (qᵍ - χ q / r). Temperate layers then drain towards ω_d, and the
# water joins the basal layer. Latent heat beyond the stored water is taken back from layer 1.
@kernel function _update_basal_water!(Hw, mᵍ, E, σgrid, H, φ, rᵇ, sᵇ, Dᵇ, qᵍ, parameters, constants, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Nσ = size(σgrid, 3)
        H̃ = column_thickness(i, j, H, parameters)
        μ = column_has_ice(i, j, H, parameters)
        L = parameters.latent_heat
        Eₚₘ = pressure_melting_enthalpy(basal_pressure(i, j, H, constants), parameters)
        q = Dᵇ[i, j, 1] * (Eₚₘ - E[i, j, 1])
        m = rᵇ[i, j, 1] * (qᵍ[i, j, 1] - grounded_part(i, j, φ) * (1 - sᵇ[i, j, 1]) * q) / (constants.ice_density * L)

        drained = zero(m)
        for k in 1:Nσ
            ω = water_fraction(E[i, j, k], layer_pressure(i, j, k, σgrid, H, constants), parameters)
            Δω = min(Δt * drainage_rate(ω, parameters), max(ω - parameters.drainage_water_fraction, zero(ω)))
            Δω = ifelse(μ, Δω, zero(Δω))
            E[i, j, k] -= L * Δω
            drained += H̃ * layer_thickness(k, σgrid) * Δω
        end

        H★ = Hw[i, j, 1] + Δt * ifelse(μ, m, zero(m)) + drained
        H⁺ = max(zero(H★), H★)
        missing_water = H⁺ - H★
        mᵍ[i, j, 1] = (H⁺ - Hw[i, j, 1]) / Δt
        Hw[i, j, 1] = H⁺
        E[i, j, 1] -= L * missing_water / (H̃ * layer_thickness(1, σgrid))
    end
end
