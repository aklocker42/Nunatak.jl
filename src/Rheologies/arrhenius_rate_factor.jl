"""
$(TYPEDEF)

Constants of the Paterson and Budd (1982, Cold Reg. Sci. Technol. 6) rate factor with the water softening of
Lliboutry and Duval (1985, Ann. Geophys. 3) as written by Greve and Blatter (2009). Defaults are the values in
PISM `pism_config.cdl` (`flow_law.Paterson_Budd.*`, `flow_law.gpbld.*`, `constants.ideal_gas_constant`).

$(TYPEDFIELDS)
"""
struct ArrheniusParameters{FT}
    "prefactor of the cold branch (Pa⁻³ s⁻¹)"
    cold_prefactor :: FT
    "prefactor of the warm branch (Pa⁻³ s⁻¹)"
    warm_prefactor :: FT
    "activation energy of the cold branch (J mol⁻¹)"
    cold_activation_energy :: FT
    "activation energy of the warm branch (J mol⁻¹)"
    warm_activation_energy :: FT
    "pressure-adjusted temperature that separates the branches (K)"
    transition_temperature :: FT
    "width of the blend between the branches, centered on `transition_temperature` (K)"
    transition_width :: FT
    "ideal gas constant `R` (J mol⁻¹ K⁻¹)"
    gas_constant :: FT
    "coefficient of the water fraction in the softening `1 + c ω`"
    water_softening_coefficient :: FT
    "water fraction above which the softening stops growing"
    maximum_water_fraction :: FT
end

"""
$(TYPEDEF)

Rate factor `A(T′, ω)` diagnosed from the enthalpy on every layer of the σ grid. `IceSheetModel` builds its
`field`, and needs `EnthalpyThermodynamics`, which update the field after each energy step.

$(TYPEDFIELDS)
"""
struct ArrheniusRateFactor{P, A}
    "`ArrheniusParameters`"
    parameters :: P
    "rate factor (Pa⁻³ s⁻¹) on the σ grid"
    field :: A
end

"""
$(TYPEDSIGNATURES)

Build an `ArrheniusRateFactor` to pass as `GlenFlowLaw(grid; rate_factor)`. The law holds for exponent `n = 3`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> ArrheniusRateFactor(grid).parameters.transition_temperature
263.15
```
"""
function ArrheniusRateFactor(grid;
                             cold_prefactor = 3.61e-13,
                             warm_prefactor = 1.73e3,
                             cold_activation_energy = 6.0e4,
                             warm_activation_energy = 13.9e4,
                             transition_temperature = 263.15,
                             transition_width = 1,
                             gas_constant = 8.31441,
                             water_softening_coefficient = 181.25,
                             maximum_water_fraction = 0.01)

    FT = eltype(grid)
    parameters = ArrheniusParameters(convert(FT, cold_prefactor), convert(FT, warm_prefactor),
                                     convert(FT, cold_activation_energy), convert(FT, warm_activation_energy),
                                     convert(FT, transition_temperature), convert(FT, transition_width),
                                     convert(FT, gas_constant), convert(FT, water_softening_coefficient),
                                     convert(FT, maximum_water_fraction))
    return ArrheniusRateFactor(parameters, nothing)
end

Base.summary(::ArrheniusRateFactor) = "ArrheniusRateFactor"

"""
$(TYPEDSIGNATURES)

Rate factor `A = A₀ exp(-Q / (R T′)) (1 + c min(ω, ωₘₐₓ))` at the pressure-adjusted temperature `T′` and water
fraction `ω`. The logarithm of `A` blends between the two branches with a smoothstep, which removes the
jump of about 1% at the transition temperature and keeps the derivative continuous.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: arrhenius_rate_factor

julia> grid = RectilinearGrid(size = (1, 1), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> A = arrhenius_rate_factor(253.15, 0.0, ArrheniusRateFactor(grid).parameters);

julia> round(A, sigdigits = 3)
1.5e-25
```
"""
@inline function arrhenius_rate_factor(T′, ω, parameters)
    R = parameters.gas_constant
    log_cold = log(parameters.cold_prefactor) - parameters.cold_activation_energy / (R * T′)
    log_warm = log(parameters.warm_prefactor) - parameters.warm_activation_energy / (R * T′)
    x = clamp((T′ - parameters.transition_temperature) / parameters.transition_width + one(T′) / 2, zero(T′), one(T′))
    s = x^2 * (3 - 2x)
    softening = 1 + parameters.water_softening_coefficient * min(ω, parameters.maximum_water_fraction)
    return exp((1 - s) * log_cold + s * log_warm) * softening
end
