"""
$(TYPEDEF)

An unstructured mesh of linear (P1) triangles with the geometry the stress balance needs
precomputed per element. Elements are stored counter-clockwise, and the node→element
adjacency is a compressed sparse row (CSR) list so that kernels gather element
contributions per node without atomics.

$(TYPEDFIELDS)
"""
struct TriangleMesh{A, FT, V <: AbstractVector{FT}, M <: AbstractMatrix{FT}, I <: AbstractVector{Int}, J <: AbstractMatrix{Int}, B <: AbstractVector{Bool}}
    "architecture the arrays live on"
    architecture :: A
    "x-coordinate of each node"
    x :: V
    "y-coordinate of each node"
    y :: V
    "`3 × Nelements` node indices of each element, counter-clockwise"
    elements :: J
    "area of each element"
    area :: V
    "`3 × Nelements` x-derivative of the three basis functions of each element"
    ∂φ∂x :: M
    "`3 × Nelements` y-derivative of the three basis functions of each element"
    ∂φ∂y :: M
    "lumped area of each node, one third of the area of its elements"
    node_area :: V
    "`Nnodes + 1` offsets into `node_elements`"
    node_element_offsets :: I
    "elements around each node, in CSR order"
    node_elements :: I
    "local index (1 to 3) of the node in each of its elements, in CSR order"
    node_local_index :: I
    "whether the x-velocity of each node is constrained"
    constrained_u :: B
    "whether the y-velocity of each node is constrained"
    constrained_v :: B
    "`2 × Nfront` node pairs of the ice-front segments"
    front_segments :: J
    "`2 × Nfront` outward unit normal of each front segment"
    front_normals :: M
    "length of each front segment"
    front_lengths :: V
end

"""
$(TYPEDSIGNATURES)

Build a `TriangleMesh` on `arch` from node coordinates `x`, `y` and a `3 × Nelements` matrix of
node indices `elements`. Elements of either orientation are accepted and stored counter-clockwise.
`constrained` holds `(u, v)` node masks, each a vector or a function `(x, y) -> Bool`; the default
constrains both velocity components on every boundary node. `front_segments` is a `2 × Nfront`
matrix of boundary node pairs on which the ice-front stress acts.

```jldoctest
julia> using Oceananigans, Nunatak

julia> x = [0.0, 1.0, 1.0, 0.0]; y = [0.0, 0.0, 1.0, 1.0];

julia> mesh = TriangleMesh(CPU(); x, y, elements = [1 1; 2 3; 3 4])
TriangleMesh{Float64} on CPU with 4 nodes, 2 elements and 0 front segments

julia> mesh.area
2-element Vector{Float64}:
 0.5
 0.5
```
"""
function TriangleMesh(arch = CPU(); x, y, elements, constrained = nothing, front_segments = nothing)
    FT = eltype(x)
    x = Vector{FT}(x)
    y = Vector{FT}(y)
    elements = Matrix{Int}(elements)
    size(elements, 1) == 3 || throw(ArgumentError("elements must be a 3 × Nelements matrix"))
    length(x) == length(y) || throw(ArgumentError("x and y must have the same length"))

    Nnodes = length(x)
    Nelements = size(elements, 2)
    area = zeros(FT, Nelements)
    ∂φ∂x = zeros(FT, 3, Nelements)
    ∂φ∂y = zeros(FT, 3, Nelements)
    node_area = zeros(FT, Nnodes)

    for e in 1:Nelements
        signed_area = element_signed_area(x, y, view(elements, :, e))
        if signed_area < 0
            elements[2, e], elements[3, e] = elements[3, e], elements[2, e]
        end
        Aₑ = abs(signed_area)
        Aₑ > 0 || throw(ArgumentError("element $e is degenerate"))
        area[e] = Aₑ
        n₁, n₂, n₃ = view(elements, :, e)
        # ∇φᵢ of P1 basis: rotated opposite edge over twice the area
        ∂φ∂x[1, e] = (y[n₂] - y[n₃]) / 2Aₑ
        ∂φ∂x[2, e] = (y[n₃] - y[n₁]) / 2Aₑ
        ∂φ∂x[3, e] = (y[n₁] - y[n₂]) / 2Aₑ
        ∂φ∂y[1, e] = (x[n₃] - x[n₂]) / 2Aₑ
        ∂φ∂y[2, e] = (x[n₁] - x[n₃]) / 2Aₑ
        ∂φ∂y[3, e] = (x[n₂] - x[n₁]) / 2Aₑ
        for i in 1:3
            node_area[elements[i, e]] += Aₑ / 3
        end
    end

    node_element_offsets, node_elements, node_local_index = node_element_adjacency(elements, Nnodes)
    boundary = boundary_nodes(elements, Nnodes)
    constrained_u, constrained_v = constraint_masks(constrained, boundary, x, y)
    front_segments, front_normals, front_lengths = front_geometry(front_segments, x, y, elements, FT)

    return TriangleMesh(arch,
                        on_architecture(arch, x), on_architecture(arch, y), on_architecture(arch, elements),
                        on_architecture(arch, area), on_architecture(arch, ∂φ∂x), on_architecture(arch, ∂φ∂y),
                        on_architecture(arch, node_area),
                        on_architecture(arch, node_element_offsets), on_architecture(arch, node_elements),
                        on_architecture(arch, node_local_index),
                        on_architecture(arch, constrained_u), on_architecture(arch, constrained_v),
                        on_architecture(arch, front_segments), on_architecture(arch, front_normals),
                        on_architecture(arch, front_lengths))
end

function element_signed_area(x, y, nodes)
    n₁, n₂, n₃ = nodes
    return ((x[n₂] - x[n₁]) * (y[n₃] - y[n₁]) - (x[n₃] - x[n₁]) * (y[n₂] - y[n₁])) / 2
end

function node_element_adjacency(elements, Nnodes)
    counts = zeros(Int, Nnodes)
    for n in elements
        counts[n] += 1
    end
    offsets = ones(Int, Nnodes + 1)
    for n in 1:Nnodes
        offsets[n + 1] = offsets[n] + counts[n]
    end
    node_elements = zeros(Int, offsets[end] - 1)
    node_local_index = zeros(Int, offsets[end] - 1)
    fill!(counts, 0)
    for e in axes(elements, 2), i in 1:3
        n = elements[i, e]
        position = offsets[n] + counts[n]
        node_elements[position] = e
        node_local_index[position] = i
        counts[n] += 1
    end
    return offsets, node_elements, node_local_index
end

# Nodes of edges that belong to one element only
function boundary_nodes(elements, Nnodes)
    edge_count = Dict{Tuple{Int, Int}, Int}()
    for e in axes(elements, 2), i in 1:3
        edge = minmax(elements[i, e], elements[mod1(i + 1, 3), e])
        edge_count[edge] = get(edge_count, edge, 0) + 1
    end
    boundary = zeros(Bool, Nnodes)
    for (edge, count) in edge_count
        count == 1 && (boundary[edge[1]] = true; boundary[edge[2]] = true)
    end
    return boundary
end

constraint_masks(::Nothing, boundary, x, y) = (copy(boundary), copy(boundary))
constraint_masks(constrained::Tuple, boundary, x, y) = (constraint_mask(constrained[1], x, y), constraint_mask(constrained[2], x, y))
constraint_mask(mask::AbstractVector, x, y) = Vector{Bool}(mask)
constraint_mask(f::Function, x, y) = Bool[f(x[n], y[n]) for n in eachindex(x)]

front_geometry(::Nothing, x, y, elements, FT) = (zeros(Int, 2, 0), zeros(FT, 2, 0), zeros(FT, 0))

function front_geometry(segments, x, y, elements, FT)
    segments = Matrix{Int}(segments)
    size(segments, 1) == 2 || throw(ArgumentError("front_segments must be a 2 × Nfront matrix"))
    Nfront = size(segments, 2)
    normals = zeros(FT, 2, Nfront)
    lengths = zeros(FT, Nfront)
    for s in 1:Nfront
        n₁, n₂ = view(segments, :, s)
        tx, ty = x[n₂] - x[n₁], y[n₂] - y[n₁]
        lengths[s] = hypot(tx, ty)
        nx, ny = ty / lengths[s], -tx / lengths[s]
        # orient the normal away from the element that owns the segment
        e = owning_element(segments[1, s], segments[2, s], elements)
        n₃ = other_node(segments[1, s], segments[2, s], view(elements, :, e))
        outward = nx * (x[n₃] - x[n₁]) + ny * (y[n₃] - y[n₁]) < 0
        normals[1, s] = ifelse(outward, nx, -nx)
        normals[2, s] = ifelse(outward, ny, -ny)
    end
    return segments, normals, lengths
end

function owning_element(n₁, n₂, elements)
    for e in axes(elements, 2)
        nodes = view(elements, :, e)
        n₁ in nodes && n₂ in nodes && return e
    end
    throw(ArgumentError("front segment ($n₁, $n₂) is not an element edge"))
end

other_node(n₁, n₂, nodes) = nodes[findfirst(n -> n != n₁ && n != n₂, nodes)]

architecture(mesh::TriangleMesh) = mesh.architecture
Base.eltype(::Type{<:TriangleMesh{A, FT}}) where {A, FT} = FT

Adapt.adapt_structure(to, mesh::TriangleMesh) =
    TriangleMesh(nothing, adapt(to, mesh.x), adapt(to, mesh.y), adapt(to, mesh.elements),
                 adapt(to, mesh.area), adapt(to, mesh.∂φ∂x), adapt(to, mesh.∂φ∂y), adapt(to, mesh.node_area),
                 adapt(to, mesh.node_element_offsets), adapt(to, mesh.node_elements), adapt(to, mesh.node_local_index),
                 adapt(to, mesh.constrained_u), adapt(to, mesh.constrained_v),
                 adapt(to, mesh.front_segments), adapt(to, mesh.front_normals), adapt(to, mesh.front_lengths))

Base.summary(mesh::TriangleMesh) =
    string("TriangleMesh{", eltype(mesh), "} on ", typeof(mesh.architecture).name.name, " with ", length(mesh.x), " nodes, ",
           size(mesh.elements, 2), " elements and ", size(mesh.front_segments, 2), " front segments")

Base.show(io::IO, mesh::TriangleMesh) = print(io, summary(mesh))

"""
$(TYPEDSIGNATURES)

Launch `kernel!` over `Nwork` linear indices of `mesh` on `arch` with `args` moved to the device.
"""
function launch!(arch, mesh::TriangleMesh, Nwork::Int, kernel!, args...)
    Nwork > 0 || return nothing
    # a static workgroup and dynamic range keep the kernel type independent of Nwork, so the launch does not allocate
    loop! = kernel!(device(arch), 256)
    loop!(convert_to_device(arch, args)...; ndrange = Nwork)
    return nothing
end
