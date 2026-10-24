"""
$(TYPEDEF)

The pair of depth-averaged velocity components `(u, v)` at `(Face, Center)` and
`(Center, Face)`, treated as one vector by the velocity solvers so that the C-grid
locations are kept.

$(TYPEDFIELDS)
"""
struct VelocityVector{U, V}
    "x-component at `(Face, Center, Nothing)`"
    u :: U
    "y-component at `(Center, Face, Nothing)`"
    v :: V
end

"""
$(TYPEDSIGNATURES)

Return a zero `VelocityVector` on `grid`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 3), x = (0, 1), y = (0, 1), topology = (Periodic, Bounded, Flat));

julia> x = VelocityVector(grid);

julia> size(x.u), size(x.v)
((4, 3, 1), (4, 4, 1))
```
"""
VelocityVector(grid) = VelocityVector(Field{Face, Center, Nothing}(grid), Field{Center, Face, Nothing}(grid))

Base.similar(x::VelocityVector) = VelocityVector(similar(x.u), similar(x.v))

Adapt.adapt_structure(to, x::VelocityVector) = VelocityVector(adapt(to, x.u), adapt(to, x.v))

Oceananigans.fill_halo_regions!(x::VelocityVector, args...; kwargs...) =
    fill_halo_regions!((x.u, x.v), args...; kwargs...)

"""
$(TYPEDSIGNATURES)

`z ← a x + b y` over the `Nx × Ny` cells of each component. On `Bounded` grids the outer wall
faces are boundary values, not unknowns, and are left untouched.
"""
function linear_combination!(z::VelocityVector, a, x::VelocityVector, b, y::VelocityVector)
    linear_combination!(z.u, a, x.u, b, y.u)
    linear_combination!(z.v, a, x.v, b, y.v)
    return z
end

function linear_combination!(z, a, x, b, y)
    launch!(architecture(z.grid), z.grid, :xyz, _linear_combination!, z, a, x, b, y)
    return z
end

@kernel function _linear_combination!(z, a, x, b, y)
    i, j, k = @index(Global, NTuple)
    @inbounds z[i, j, k] = a * x[i, j, k] + b * y[i, j, k]
end

"""
$(TYPEDSIGNATURES)

Set `x` to `y` over the interior.
"""
Base.copyto!(x::VelocityVector, y::VelocityVector) = linear_combination!(x, one(eltype(y.u)), y, zero(eltype(y.u)), y)

function Base.fill!(x::VelocityVector, value)
    interior(x.u) .= value
    interior(x.v) .= value
    return x
end

"""
$(TYPEDSIGNATURES)

`x ← x + α y` over the interior.
"""
axpy!(α, y::VelocityVector, x::VelocityVector) = linear_combination!(x, one(α), x, α, y)

"""
$(TYPEDSIGNATURES)

`x ← y + β x` over the interior.
"""
xpby!(y::VelocityVector, β, x::VelocityVector) = linear_combination!(x, one(β), y, β, x)

"""
$(TYPEDSIGNATURES)

`r ← b − q` over the interior.
"""
subtract!(r::VelocityVector, b::VelocityVector, q::VelocityVector) = linear_combination!(r, one(eltype(b.u)), b, -one(eltype(b.u)), q)

"""
$(TYPEDSIGNATURES)

`z ← x / y` elementwise over the interior.
"""
function divide!(z::VelocityVector, x::VelocityVector, y::VelocityVector)
    divide!(z.u, x.u, y.u)
    divide!(z.v, x.v, y.v)
    return z
end

function divide!(z, x, y)
    launch!(architecture(z.grid), z.grid, :xyz, _divide!, z, x, y)
    return z
end

@kernel function _divide!(z, x, y)
    i, j, k = @index(Global, NTuple)
    @inbounds z[i, j, k] = x[i, j, k] / y[i, j, k]
end

"""
$(TYPEDSIGNATURES)

Dot product over the `Nx × Ny` cells of each component, via the elementwise product in the
workspace `w`. The halos and wall faces of `w` must be zero, as they are for `VelocityVector(grid)`.
"""
function vector_dot!(w::VelocityVector, x::VelocityVector, y::VelocityVector)
    return component_dot!(w.u, x.u, y.u) + component_dot!(w.v, x.v, y.v)
end

function component_dot!(w, x, y)
    launch!(architecture(w.grid), w.grid, :xyz, _product!, w, x, y)
    # halos and wall faces of w stay zero, so the parent sum is the interior sum
    return sum(parent(w))
end

@kernel function _product!(w, x, y)
    i, j, k = @index(Global, NTuple)
    @inbounds w[i, j, k] = x[i, j, k] * y[i, j, k]
end

LinearAlgebra.dot(x::VelocityVector, y::VelocityVector) = vector_dot!(VelocityVector(x.u.grid), x, y)
LinearAlgebra.norm(x::VelocityVector) = sqrt(LinearAlgebra.dot(x, x))
