"""
$(TYPEDEF)

Physical constants shared by the driving stress, the flotation condition and the effective pressure.

$(TYPEDFIELDS)
"""
struct IceSheetConstants{FT}
    "ice density `ρᵢ` (kg m⁻³)"
    ice_density :: FT
    "ocean density `ρw` (kg m⁻³)"
    ocean_density :: FT
    "gravitational acceleration `g` (m s⁻²)"
    gravitational_acceleration :: FT
end

"""
$(TYPEDSIGNATURES)

Build `IceSheetConstants` with the floating-point type of `grid`. The default densities are those of MISMIP+.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> constants = IceSheetConstants(grid);

julia> constants.ocean_density / constants.ice_density
1.1198257080610021
```
"""
function IceSheetConstants(grid::AbstractGrid; ice_density = 918, ocean_density = 1028, gravitational_acceleration = 9.81)
    FT = eltype(grid)
    return IceSheetConstants(convert(FT, ice_density), convert(FT, ocean_density), convert(FT, gravitational_acceleration))
end

Base.summary(c::IceSheetConstants) = string("IceSheetConstants(ρᵢ = ", c.ice_density, ", ρw = ", c.ocean_density,
                                            ", g = ", c.gravitational_acceleration, ")")
