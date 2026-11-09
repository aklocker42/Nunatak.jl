"""
$(TYPEDEF)

Depth-integrated stress balance. `shear = nothing` gives the shallow-shelf approximation (SSA);
`shear = DIVAShear()` adds vertical shear through the column integral `F₂` (DIVA). `side_walls` sets the
tangential velocity condition on `Bounded` walls, `FreeSlipWalls()` or `NoSlipWalls()`. Both share
one velocity operator that takes the depth-integrated viscosity `η̄H` and the effective drag
coefficient `β_eff = β / (1 + β F₂)`; SSA passes `F₂ = 0`.

$(TYPEDFIELDS)
"""
struct DepthIntegrated{S, W}
    "vertical-shear component: `nothing` (SSA) or `DIVAShear()`"
    shear :: S
    "tangential condition on `Bounded` walls: `FreeSlipWalls()` or `NoSlipWalls()`"
    side_walls :: W
end

"""
$(TYPEDSIGNATURES)

```jldoctest
julia> using Nunatak

julia> DepthIntegrated(side_walls = NoSlipWalls())
DepthIntegrated{Nothing, NoSlipWalls}(nothing, NoSlipWalls())
```
"""
DepthIntegrated(; shear = nothing, side_walls = FreeSlipWalls()) = DepthIntegrated(shear, side_walls)

"""
$(TYPEDEF)

Tangential stress-free condition on `Bounded` walls: the shear strain rate vanishes at wall corners.
"""
struct FreeSlipWalls end

"""
$(TYPEDEF)

Tangential no-slip condition on `Bounded` walls: the velocity along the wall is zero at the wall, imposed
with a mirror ghost so the shear strain rate at a wall corner is the one-sided `u / (Δy / 2) / 2`.
"""
struct NoSlipWalls end

"""
$(TYPEDEF)

Vertical-shear component of the depth-integrated viscous approximation (Goldberg 2011;
Lipscomb et al. 2019): the viscosity includes the vertical shear driven by the basal stress, the
depth-averaged velocity feels the drag `β / (1 + β F₂)`, and the velocity profile is reconstructed from the
shear integral after each solve.
"""
struct DIVAShear end

const SSA  = DepthIntegrated{Nothing}
const DIVA = DepthIntegrated{<:DIVAShear}

Base.summary(::SSA) = "DepthIntegrated (SSA)"
Base.summary(::DIVA) = "DepthIntegrated (DIVA)"
