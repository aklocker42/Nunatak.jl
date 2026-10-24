"""
$(TYPEDEF)

Depth-integrated stress balance. `shear = nothing` gives the shallow-shelf approximation (SSA);
`shear = DIVAShear()` adds vertical shear through the column integral `F₂` (DIVA). Both share
one velocity operator that takes the depth-integrated viscosity `η̄H` and the effective drag
coefficient `β_eff = β / (1 + β F₂)`; SSA passes `F₂ = 0`.

$(TYPEDFIELDS)
"""
struct DepthIntegrated{S}
    "vertical-shear component: `nothing` (SSA) or `DIVAShear()`"
    shear :: S
end

"""
$(TYPEDSIGNATURES)

```jldoctest
julia> using Nunatak

julia> DepthIntegrated()
DepthIntegrated{Nothing}(nothing)
```
"""
DepthIntegrated(; shear = nothing) = DepthIntegrated(shear)

"""
$(TYPEDEF)

Vertical-shear component of the depth-integrated viscous approximation (Goldberg 2011;
Lipscomb et al. 2019). Switched on in phase 5; the layout it needs exists from phase 0.
"""
struct DIVAShear end

const SSA  = DepthIntegrated{Nothing}
const DIVA = DepthIntegrated{<:DIVAShear}

Base.summary(::SSA) = "DepthIntegrated (SSA)"
Base.summary(::DIVA) = "DepthIntegrated (DIVA)"
