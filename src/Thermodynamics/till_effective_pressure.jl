"""
$(TYPEDEF)

Effective pressure of a saturated till (Bueler and van Pelt, 2015, Geosci. Model Dev. 8, 1613–1635), from the
basal water `Hw` of [`EnthalpyThermodynamics`](@ref):

    N = min(N₀ (δ P₀ / N₀)ˢ 10^((e₀ / Cc) (1 - s)), Nᵒ),    s = min(Hw / Wᵐᵃˣ, 1),

with the overburden `P₀ = ρᵢ g H`. A dry till carries `N₀ 10^(e₀ / Cc)`, a saturated till `δ P₀`.
`Nᵒ = max(P₀ + ρw g min(b, 0), 0)` is the effective pressure of a bed connected to the ocean, so `N` still
vanishes at flotation. The till drains at `drainage_rate` and holds at most `maximum_water_thickness`;
water beyond it leaves the column.

$(TYPEDFIELDS)
"""
struct TillEffectivePressure{FT}
    "effective pressure `N₀` at the reference void ratio (Pa)"
    reference_effective_pressure :: FT
    "effective pressure of a saturated till as a fraction `δ` of the overburden"
    overburden_fraction :: FT
    "void ratio `e₀` at `N₀`"
    reference_void_ratio :: FT
    "compressibility `Cc` of the till"
    compressibility :: FT
    "water thickness `Wᵐᵃˣ` that saturates the till (m)"
    maximum_water_thickness :: FT
    "rate at which the till water drains (m s⁻¹)"
    drainage_rate :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `TillEffectivePressure` with the default parameters of Bueler and van Pelt (2015) and a drainage
of 1 mm per year.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> TillEffectivePressure(grid)
TillEffectivePressure(reference_effective_pressure = 1000.0, overburden_fraction = 0.02, maximum_water_thickness = 2.0)
```
"""
function TillEffectivePressure(grid;
                               reference_effective_pressure = 1000,
                               overburden_fraction = 0.02,
                               reference_void_ratio = 0.69,
                               compressibility = 0.12,
                               maximum_water_thickness = 2,
                               drainage_rate = 1e-3 / (365.25 * 86400))
    FT = eltype(grid)
    return TillEffectivePressure(convert(FT, reference_effective_pressure), convert(FT, overburden_fraction),
                                 convert(FT, reference_void_ratio), convert(FT, compressibility),
                                 convert(FT, maximum_water_thickness), convert(FT, drainage_rate))
end

Base.summary(law::TillEffectivePressure) =
    string("TillEffectivePressure(reference_effective_pressure = ", law.reference_effective_pressure,
           ", overburden_fraction = ", law.overburden_fraction,
           ", maximum_water_thickness = ", law.maximum_water_thickness, ")")

Base.show(io::IO, law::TillEffectivePressure) = print(io, summary(law))

@inline effective_pressure(i, j, P₀, Nᵒ, Hw, ::Nothing) = Nᵒ

# A positive stand-in for `P₀ = 0` keeps the power differentiable; `Nᵒ = 0` there sets `N`.
@inline function effective_pressure(i, j, P₀, Nᵒ, Hw, law::TillEffectivePressure)
    N₀, δ = law.reference_effective_pressure, law.overburden_fraction
    s = @inbounds clamp(Hw[i, j, 1] / law.maximum_water_thickness, zero(P₀), one(P₀))
    P̃ = ifelse(P₀ > 0, P₀, N₀ / δ)
    Nᵗ = N₀ * (δ * P̃ / N₀)^s * 10^(law.reference_void_ratio / law.compressibility * (1 - s))
    return min(Nᵗ, Nᵒ)
end

@inline drain_till(Hw, ::Nothing, Δt) = Hw
@inline drain_till(Hw, law::TillEffectivePressure, Δt) = clamp(Hw - Δt * law.drainage_rate, zero(Hw), law.maximum_water_thickness)
