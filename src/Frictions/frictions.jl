"""
$(TYPEDEF)

Weertman sliding law `τ_b = C |u_b|^(m-1) u_b`, evaluated with the basal velocity `u_b`.

$(TYPEDFIELDS)
"""
struct WeertmanFriction{C, FT}
    "friction coefficient `C`, a `Field` at `(Center, Center, Nothing)`"
    coefficient :: C
    "sliding exponent `m`"
    exponent :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `WeertmanFriction` on `grid`; `coefficient` is required.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> friction = WeertmanFriction(grid; coefficient = 3e4);

julia> friction.exponent
0.3333333333333333
```
"""
function WeertmanFriction(grid; coefficient, exponent = 1/3)
    FT = eltype(grid)
    return WeertmanFriction(parameter_field(grid, coefficient), convert(FT, exponent))
end

"""
$(TYPEDEF)

Regularised Coulomb sliding law (Tsai et al. 2015), `τ_b = min(C |u_b|^m, α² N) u_b / |u_b|`,
with a smooth minimum of width `smoothing` so that the law is differentiable.

$(TYPEDFIELDS)
"""
struct RegularizedCoulombFriction{C, FT}
    "friction coefficient `C`, a `Field` at `(Center, Center, Nothing)`"
    coefficient :: C
    "sliding exponent `m`"
    exponent :: FT
    "Coulomb coefficient `α²`"
    coulomb_coefficient :: FT
    "width of the smooth minimum (Pa)"
    smoothing :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `RegularizedCoulombFriction` on `grid`; `coefficient`, `coulomb_coefficient` and
`smoothing` are required.
"""
function RegularizedCoulombFriction(grid; coefficient, coulomb_coefficient, smoothing, exponent = 1/3)
    FT = eltype(grid)
    return RegularizedCoulombFriction(parameter_field(grid, coefficient),
                                      convert(FT, exponent),
                                      convert(FT, coulomb_coefficient),
                                      convert(FT, smoothing))
end

Base.summary(f::WeertmanFriction) = string("WeertmanFriction(exponent = ", f.exponent, ")")
Base.summary(f::RegularizedCoulombFriction) = string("RegularizedCoulombFriction(exponent = ", f.exponent, ")")
