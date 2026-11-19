"""
$(TYPEDEF)

Location marker of a `MeshField` stored at the nodes of a `TriangleMesh`.
"""
struct Node end

"""
$(TYPEDEF)

Location marker of a `MeshField` stored per element of a `TriangleMesh`.
"""
struct Element end

"""
$(TYPEDEF)

A scalar field on a `TriangleMesh`, one value per node or per element.

$(TYPEDFIELDS)
"""
struct MeshField{L, D, M}
    "device vector of values"
    data :: D
    "the mesh"
    mesh :: M
    MeshField{L}(data::D, mesh::M) where {L, D, M} = new{L, D, M}(data, mesh)
end

const NodeField = MeshField{Node}
const ElementField = MeshField{Element}

"""
$(TYPEDSIGNATURES)

Return a zero field at the nodes of `mesh`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> mesh = TriangleMesh(CPU(); x = [0.0, 1.0, 1.0, 0.0], y = [0.0, 0.0, 1.0, 1.0], elements = [1 1; 2 3; 3 4]);

julia> f = NodeField(mesh); set!(f, (x, y) -> x + 2y); interior(f)
4-element Vector{Float64}:
 0.0
 1.0
 3.0
 2.0
```
"""
NodeField(mesh::TriangleMesh) = MeshField{Node}(zero(mesh.node_area), mesh)

"""
$(TYPEDSIGNATURES)

Return a zero field on the elements of `mesh`.
"""
ElementField(mesh::TriangleMesh) = MeshField{Element}(zero(mesh.area), mesh)

Base.parent(f::MeshField) = f.data
Oceananigans.Fields.interior(f::MeshField) = f.data
Base.similar(f::MeshField{L}) where L = MeshField{L}(similar(f.data), f.mesh)
Base.eltype(f::MeshField) = eltype(f.data)
Base.length(f::MeshField) = length(f.data)
Adapt.adapt_structure(to, f::MeshField{L}) where L = MeshField{L}(adapt(to, f.data), nothing)
architecture(f::MeshField) = architecture(f.mesh)

Base.summary(f::MeshField{L}) where L = string(length(f), "-point ", L, " MeshField on ", summary(f.mesh))
Base.show(io::IO, f::MeshField) = print(io, summary(f))

"""
$(TYPEDSIGNATURES)

Set `f` to a number, a vector or a function `(x, y)` of the node coordinates.
"""
set!(f::MeshField, value::Number) = (fill!(f.data, value); return f)
set!(f::MeshField, value::AbstractArray) = (copyto!(f.data, value); return f)

function set!(f::NodeField, value::Function)
    x = on_architecture(CPU(), f.mesh.x)
    y = on_architecture(CPU(), f.mesh.y)
    copyto!(f.data, [convert(eltype(f), value(x[n], y[n])) for n in eachindex(x)])
    return f
end

function set!(f::ElementField, value::Function)
    x = on_architecture(CPU(), f.mesh.x)
    y = on_architecture(CPU(), f.mesh.y)
    elements = on_architecture(CPU(), f.mesh.elements)
    values = [convert(eltype(f), value(sum(x[elements[i, e]] for i in 1:3) / 3,
                                        sum(y[elements[i, e]] for i in 1:3) / 3)) for e in axes(elements, 2)]
    copyto!(f.data, values)
    return f
end

fill_halo_regions!(::MeshField, args...; kwargs...) = nothing
fill_halo_regions!(::NTuple{N, MeshField}, args...; kwargs...) where N = nothing

parameter_field(mesh::TriangleMesh, value, LX = Center, LY = Center) = set!(NodeField(mesh), value)
parameter_field(mesh::TriangleMesh, value::MeshField, LX = Center, LY = Center) = value
parameter_field(mesh::TriangleMesh, ::Nothing, LX = Center, LY = Center) = nothing

"""
$(TYPEDSIGNATURES)

Return a zero `VelocityVector` of two `NodeField`s on `mesh`.
"""
VelocityVector(mesh::TriangleMesh) = VelocityVector(NodeField(mesh), NodeField(mesh))

function linear_combination!(z::MeshField, a, x::MeshField, b, y::MeshField)
    launch!(architecture(z), z.mesh, length(z), _mesh_linear_combination!, z, a, x, b, y)
    return z
end

@kernel function _mesh_linear_combination!(z, a, x, b, y)
    n = @index(Global, Linear)
    @inbounds z.data[n] = a * x.data[n] + b * y.data[n]
end

function divide!(z::MeshField, x::MeshField, y::MeshField)
    launch!(architecture(z), z.mesh, length(z), _mesh_divide!, z, x, y)
    return z
end

@kernel function _mesh_divide!(z, x, y)
    n = @index(Global, Linear)
    @inbounds z.data[n] = x.data[n] / y.data[n]
end

function component_dot!(w::MeshField, x::MeshField, y::MeshField)
    launch!(architecture(w), w.mesh, length(w), _mesh_product!, w, x, y)
    return sum(w.data)
end

@kernel function _mesh_product!(w, x, y)
    n = @index(Global, Linear)
    @inbounds w.data[n] = x.data[n] * y.data[n]
end

LinearAlgebra.dot(x::VelocityVector{<:MeshField}, y::VelocityVector{<:MeshField}) = vector_dot!(VelocityVector(x.u.mesh), x, y)
