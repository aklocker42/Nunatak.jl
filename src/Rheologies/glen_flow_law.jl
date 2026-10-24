"""
$(TYPEDEF)

Glen's flow law `η = ½ A^(-1/n) ε̇ₑ^((1-n)/n)` with the regularised effective strain rate
`ε̇ₑ² = ... + ε̇₀²`. The rate factor `A` is a `Field` so that gradients with respect to it exist.

$(TYPEDFIELDS)
"""
struct GlenFlowLaw{A, FT}
    "rate factor `A` (Pa⁻ⁿ s⁻¹), a `Field` at `(Center, Center, Nothing)`"
    rate_factor :: A
    "flow-law exponent `n`"
    exponent :: FT
    "strain-rate regularisation `ε̇₀` (s⁻¹)"
    strain_rate_regularization :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `GlenFlowLaw` on `grid`. `rate_factor` is required and may be a number, an array,
a function `(x, y)`, or a `Field`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> rheology = GlenFlowLaw(grid; rate_factor = 1e-24);

julia> rheology.exponent
3.0
```
"""
function GlenFlowLaw(grid; rate_factor, exponent = 3, strain_rate_regularization = 1e-13)
    FT = eltype(grid)
    A = parameter_field(grid, rate_factor)
    return GlenFlowLaw(A, convert(FT, exponent), convert(FT, strain_rate_regularization))
end

Base.summary(r::GlenFlowLaw) = string("GlenFlowLaw(exponent = ", r.exponent, ")")
