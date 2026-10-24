"""
$(TYPEDEF)

Sub-grid grounded fraction `φ ∈ [0, 1]` from the flotation function `f = H + ρ_w b / ρ_i`,
smoothed over `smoothing_thickness` so that `φ` is differentiable; the hard step is the limit
of vanishing smoothing. Friction is scaled by `φ` and melt by `1 - φ`.

$(TYPEDFIELDS)
"""
struct SubgridGroundedFraction{FT}
    "flotation-function smoothing thickness `f₀` (m)"
    smoothing_thickness :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `SubgridGroundedFraction` with the floating-point type of `grid`.
"""
SubgridGroundedFraction(grid::Oceananigans.Grids.AbstractGrid; smoothing_thickness) = SubgridGroundedFraction(convert(eltype(grid), smoothing_thickness))
