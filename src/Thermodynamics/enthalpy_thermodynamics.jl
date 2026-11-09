"""
$(TYPEDEF)

Material constants and regularization widths of the enthalpy formulation, passed to the kernels.
Defaults follow PISM `pism_config.cdl` and Table A1 of Kleiner et al. (2015, The Cryosphere 9).

$(TYPEDFIELDS)
"""
struct EnthalpyParameters{FT}
    "specific heat capacity of ice `cᵢ` (J kg⁻¹ K⁻¹)"
    heat_capacity :: FT
    "thermal conductivity of ice `kᵢ` (W m⁻¹ K⁻¹)"
    thermal_conductivity :: FT
    "latent heat of fusion `L` (J kg⁻¹)"
    latent_heat :: FT
    "Clausius-Clapeyron constant `β` (K Pa⁻¹)"
    clausius_clapeyron :: FT
    "melting temperature at zero pressure `T₀` (K)"
    melting_temperature :: FT
    "temperature at which the enthalpy is zero `Tᵣ` (K)"
    reference_temperature :: FT
    "ratio of temperate to cold enthalpy diffusivity"
    temperate_diffusivity_ratio :: FT
    "enthalpy width of the cold-temperate blend and of the basal switch (J kg⁻¹)"
    enthalpy_blend_width :: FT
    "basal water thickness over which the base becomes wet (m)"
    water_layer_blend_thickness :: FT
    "columns thinner than this are ice free and hold the surface enthalpy (m)"
    minimum_thickness :: FT
    "water fraction above which temperate ice drains to the bed"
    drainage_water_fraction :: FT
    "largest drainage rate of the water fraction (s⁻¹), reached at three times the drainage water fraction"
    maximum_drainage_rate :: FT
end

"""
$(TYPEDEF)

Enthalpy thermodynamics after Aschwanden et al. (2012): one prognostic enthalpy `E` per layer of the
σ grid covers cold ice (`E < Eₚₘ`) and temperate ice with water fraction `ω`. Basal melt water is stored
in a local layer `Hw` without transport, and refreezes when the column draws heat from the base.
Water above `drainage_water_fraction` drains from temperate layers into `Hw`. A temperate base under a
temperate layer is insulating (Kleiner et al., 2015, Eq. 6), so all basal heat melts ice there. With
`grounded_melt_coupling` the change of `Hw` is taken from the ice thickness, so ice and water mass is conserved.
Horizontal advection is explicit and first-order upwind, so an energy step needs `u Δt / Δx ≤ 1`.
Stepping the energy on a sparse `schedule` counts the elapsed time in that limit.

$(TYPEDFIELDS)
"""
struct EnthalpyThermodynamics{P, G, TS, TO, SC, ST, LT}
    "`EnthalpyParameters`"
    parameters :: P
    "geothermal heat flux into the base `G` (W m⁻²)"
    geothermal_flux :: G
    "surface temperature `Tₛ` (K), capped at the melting temperature"
    surface_temperature :: TS
    "temperature of the ocean at the base of floating ice (K)"
    ocean_temperature :: TO
    "schedule of the energy steps; the elapsed time since the last step is the time step"
    schedule :: SC
    "fields and column solver, built by `IceSheetModel`"
    state :: ST
    "time of the last energy step"
    last_step_time :: LT
    "whether grounded melt and drainage thin the ice"
    grounded_melt_coupling :: Bool
end

"""
$(TYPEDSIGNATURES)

Build `EnthalpyThermodynamics` for an `IceSheetModel` on `grid`. `geothermal_flux`, `surface_temperature`
and `ocean_temperature` may be a number, an array, a function `(x, y)` or a `Field`. The energy equation is
solved once per time step by default; `schedule = IterationInterval(N)` steps it every `N` ice steps.
The enthalpy starts at the surface value; set it with `set!(model; E)` or `set!(model; T)`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), halo = (4, 4), x = (0, 1e5), y = (0, 1e5), topology = (Bounded, Bounded, Flat));

julia> thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux = 0.042, surface_temperature = 243.15);

julia> thermodynamics.parameters.latent_heat
334000.0
```
"""
function EnthalpyThermodynamics(grid;
                                geothermal_flux = 0,
                                surface_temperature = 243.15,
                                ocean_temperature = 271.15,
                                heat_capacity = 2009,
                                thermal_conductivity = 2.1,
                                latent_heat = 3.34e5,
                                clausius_clapeyron = 7.9e-8,
                                melting_temperature = 273.15,
                                reference_temperature = 223.15,
                                temperate_diffusivity_ratio = 1e-3,
                                enthalpy_blend_width = 1,
                                water_layer_blend_thickness = 1e-3,
                                minimum_thickness = 1,
                                drainage_water_fraction = 0.01,
                                maximum_drainage_rate = 1.58443823077064e-9,
                                grounded_melt_coupling = true,
                                schedule = IterationInterval(1))

    FT = eltype(grid)
    parameters = EnthalpyParameters(convert(FT, heat_capacity), convert(FT, thermal_conductivity),
                                    convert(FT, latent_heat), convert(FT, clausius_clapeyron),
                                    convert(FT, melting_temperature), convert(FT, reference_temperature),
                                    convert(FT, temperate_diffusivity_ratio), convert(FT, enthalpy_blend_width),
                                    convert(FT, water_layer_blend_thickness), convert(FT, minimum_thickness),
                                    convert(FT, drainage_water_fraction), convert(FT, maximum_drainage_rate))

    return EnthalpyThermodynamics(parameters,
                                  parameter_field(grid, geothermal_flux),
                                  parameter_field(grid, surface_temperature),
                                  parameter_field(grid, ocean_temperature),
                                  schedule, nothing, Ref(zero(FT)), grounded_melt_coupling)
end

Base.summary(th::EnthalpyThermodynamics) = string("EnthalpyThermodynamics(schedule = ", summary(th.schedule), ")")

"""
$(TYPEDSIGNATURES)

Build the fields and the column solver of `thermodynamics` on `σgrid`, and start the enthalpy at the surface value.
"""
function materialize_thermodynamics(thermodynamics::EnthalpyThermodynamics, grid, σgrid, clock)
    FT = eltype(grid)

    E = CenterField(σgrid)
    T = CenterField(σgrid)
    ω = CenterField(σgrid)
    lower_diagonal = CenterField(σgrid)
    diagonal = CenterField(σgrid)
    upper_diagonal = CenterField(σgrid)
    right_hand_side = CenterField(σgrid)
    column_solver = BatchedTridiagonalSolver(σgrid; lower_diagonal, diagonal, upper_diagonal)

    Hw = Field{Center, Center, Nothing}(grid)
    mᵍ = Field{Center, Center, Nothing}(grid)
    rᵇ = Field{Center, Center, Nothing}(grid)
    sᵇ = Field{Center, Center, Nothing}(grid)
    Dᵇ = Field{Center, Center, Nothing}(grid)
    qᵍ = Field{Center, Center, Nothing}(grid)
    Hⁿ = Field{Center, Center, Nothing}(grid)
    W = ZFaceField(σgrid)
    Φ = CenterField(σgrid)

    state = (; E, T, ω, Hw, mᵍ, rᵇ, sᵇ, Dᵇ, qᵍ, Hⁿ, W, Φ, right_hand_side, column_solver)

    (; parameters, geothermal_flux, surface_temperature, ocean_temperature, schedule, grounded_melt_coupling) = thermodynamics
    materialized = EnthalpyThermodynamics(parameters, geothermal_flux, surface_temperature, ocean_temperature,
                                          schedule, state, Ref(convert(FT, clock.time)), grounded_melt_coupling)

    launch!(architecture(σgrid), σgrid, :xyz, _set_surface_enthalpy!, E, surface_temperature, parameters)
    return materialized
end

materialize_thermodynamics(::Nothing, grid, σgrid, clock) = nothing

@kernel function _set_surface_enthalpy!(E, Tₛ, parameters)
    i, j, k = @index(Global, NTuple)
    @inbounds E[i, j, k] = surface_enthalpy(i, j, Tₛ, parameters)
end

prognostic_state(th::EnthalpyThermodynamics) = (E = prognostic_state(th.state.E),
                                                Hw = prognostic_state(th.state.Hw),
                                                Hⁿ = prognostic_state(th.state.Hⁿ),
                                                mᵍ = prognostic_state(th.state.mᵍ),
                                                last_step_time = th.last_step_time[])

function restore_prognostic_state!(restored::EnthalpyThermodynamics, from)
    restore_prognostic_state!(restored.state.E, from.E)
    restore_prognostic_state!(restored.state.Hw, from.Hw)
    restore_prognostic_state!(restored.state.Hⁿ, from.Hⁿ)
    restore_prognostic_state!(restored.state.mᵍ, from.mᵍ)
    restored.last_step_time[] = from.last_step_time
    return restored
end

restore_prognostic_state!(::EnthalpyThermodynamics, ::Nothing) = nothing
