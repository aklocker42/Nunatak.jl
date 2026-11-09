"""
$(TYPEDSIGNATURES)

Melting temperature `Tₚₘ = T₀ - β p` at pressure `p`.
"""
@inline pressure_melting_temperature(p, parameters) = parameters.melting_temperature - parameters.clausius_clapeyron * p

"""
$(TYPEDSIGNATURES)

Enthalpy of ice at its melting point, `Eₚₘ = cᵢ (Tₚₘ - Tᵣ)`, at pressure `p`.
"""
@inline pressure_melting_enthalpy(p, parameters) = parameters.heat_capacity * (pressure_melting_temperature(p, parameters) - parameters.reference_temperature)

"""
$(TYPEDSIGNATURES)

Temperature of ice with enthalpy `E` at pressure `p`: `Tᵣ + E / cᵢ` in cold ice and `Tₚₘ` in temperate ice.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: enthalpy, temperature, water_fraction

julia> grid = RectilinearGrid(size = (1, 1), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> parameters = EnthalpyThermodynamics(grid).parameters;

julia> E = enthalpy(263.15, 0, 1e6, parameters);

julia> (temperature(E, 1e6, parameters), water_fraction(E, 1e6, parameters))
(263.15, 0.0)
```
"""
@inline temperature(E, p, parameters) = min(pressure_melting_temperature(p, parameters), parameters.reference_temperature + E / parameters.heat_capacity)

"""
$(TYPEDSIGNATURES)

Liquid water fraction `ω = (E - Eₚₘ) / L` of temperate ice, zero in cold ice.
"""
@inline water_fraction(E, p, parameters) = max(zero(E), E - pressure_melting_enthalpy(p, parameters)) / parameters.latent_heat

"""
$(TYPEDSIGNATURES)

Enthalpy of ice at temperature `T` with water fraction `ω` at pressure `p`. Temperatures above `Tₚₘ` are capped at it.
"""
@inline function enthalpy(T, ω, p, parameters)
    T̃ = min(T, pressure_melting_temperature(p, parameters))
    return parameters.heat_capacity * (T̃ - parameters.reference_temperature) + ω * parameters.latent_heat
end

@inline surface_enthalpy(i, j, Tₛ, parameters) = @inbounds enthalpy(Tₛ[i, j, 1], zero(parameters.latent_heat), zero(parameters.latent_heat), parameters)

@inline column_thickness(i, j, H, parameters) = @inbounds max(H[i, j, 1], parameters.minimum_thickness)

@inline column_has_ice(i, j, H, parameters) = @inbounds H[i, j, 1] > parameters.minimum_thickness

# Pressure at the center of layer k.
@inline function layer_pressure(i, j, k, σgrid, H, constants)
    ζ = znode(1, 1, k, σgrid, Center(), Center(), Center())
    return @inbounds constants.ice_density * constants.gravitational_acceleration * H[i, j, 1] * (1 - ζ)
end

@inline basal_pressure(i, j, H, constants) = @inbounds constants.ice_density * constants.gravitational_acceleration * H[i, j, 1]

# Rises from 0 to 1 over `width` above zero.
@inline unit_ramp(x, width) = clamp(x / width, zero(x), one(x))

# Enthalpy diffusivity K = k / c in cold ice, reduced in temperate ice with a narrow linear blend below Eₚₘ.
@inline function layer_diffusivity(i, j, k, σgrid, E, H, parameters, constants)
    p = layer_pressure(i, j, k, σgrid, H, constants)
    Eₚₘ = pressure_melting_enthalpy(p, parameters)
    w = @inbounds unit_ramp(E[i, j, k] - Eₚₘ + parameters.enthalpy_blend_width, parameters.enthalpy_blend_width)
    Kᶜ = parameters.thermal_conductivity / parameters.heat_capacity
    return Kᶜ * (1 - w + w * parameters.temperate_diffusivity_ratio)
end
