"""
$(TYPEDEF)

Ice mask `μ(H)` that rises from 0 at `minimum_thickness` to 1 at `minimum_thickness + transition_thickness`
as a cubic smoothstep, so it is continuously differentiable in `H`. The depth-integrated viscosity is
scaled by `μ`, so cells with `μ = 0` are ice free and add no stiffness to the stress balance; velocity
points with ice-free cells on both sides are removed from the solve like walls and get zero driving stress.

$(TYPEDFIELDS)
"""
struct SmoothIceMask{FT}
    "thickness at and below which a cell is ice free (m)"
    minimum_thickness :: FT
    "thickness over which the mask rises from 0 to 1 (m)"
    transition_thickness :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `SmoothIceMask` with the floating-point type of `grid`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: ice_mask_value

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> masking = SmoothIceMask(grid; minimum_thickness = 1, transition_thickness = 2);

julia> [ice_mask_value(H, masking) for H in (0.5, 1, 2, 3, 10)]
5-element Vector{Float64}:
 0.0
 0.0
 0.5
 1.0
 1.0
```
"""
function SmoothIceMask(grid::AbstractGrid; minimum_thickness = 1, transition_thickness = 1)
    transition_thickness > 0 || throw(ArgumentError("transition_thickness must be positive, got $transition_thickness"))
    FT = eltype(grid)
    return SmoothIceMask(convert(FT, minimum_thickness), convert(FT, transition_thickness))
end

Base.summary(masking::SmoothIceMask) = string("SmoothIceMask(minimum_thickness = ", masking.minimum_thickness,
                                              ", transition_thickness = ", masking.transition_thickness, ")")

@inline function ice_mask_value(H, masking::SmoothIceMask)
    t = clamp((H - masking.minimum_thickness) / masking.transition_thickness, zero(H), one(H))
    return t^2 * (3 - 2t)
end

"""
$(TYPEDSIGNATURES)

Compute the ice mask `μ` at cell centres from the thickness `H`.
"""
function compute_ice_mask!(μ, masking, H)
    grid = H.grid
    launch!(architecture(grid), grid, :xy, _compute_ice_mask!, μ, masking, H)
    return nothing
end

@kernel function _compute_ice_mask!(μ, masking, H)
    i, j = @index(Global, NTuple)
    @inbounds μ[i, j, 1] = ice_mask_value(H[i, j, 1], masking)
end

# A velocity point is ice free when the non-negative mask (or stiffness) vanishes on both sides.
@inline ice_freeᶠᶜᶜ(i, j, k, grid, μ) = @inbounds iszero(μ[i-1, j, k] + μ[i, j, k])
@inline ice_freeᶜᶠᶜ(i, j, k, grid, μ) = @inbounds iszero(μ[i, j-1, k] + μ[i, j, k])
@inline ice_freeᶠᶜᶜ(i, j, k, grid, ::Nothing) = false
@inline ice_freeᶜᶠᶜ(i, j, k, grid, ::Nothing) = false

@inline ice_mask_at(i, j, k, μ) = @inbounds μ[i, j, k]
@inline ice_mask_at(i, j, k, ::Nothing) = true
