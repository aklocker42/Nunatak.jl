"""
$(TYPEDSIGNATURES)

Apply the linearized depth-integrated stress-balance operator on a `TriangleMesh` to the nodal
velocity `p` and store the result in `q`. This is the P1 Galerkin form of `ssa_operator!`:
row `i` of element `e` adds `η̄Hₑ Aₑ ((4bⱼbᵢ + cⱼcᵢ) uⱼ + (2cⱼbᵢ + bⱼcᵢ) vⱼ)` to `q.u` and
`η̄Hₑ Aₑ ((2bⱼcᵢ + cⱼbᵢ) uⱼ + (4cⱼcᵢ + bⱼbᵢ) vⱼ)` to `q.v`, with `bᵢ, cᵢ` the basis gradients and `Aₑ`
the element area, plus the lumped drag `Aₙ βₙ` times the nodal velocity, with `Aₙ` the node area.
`η̄H` is an `ElementField` and `β` a `NodeField`; the trailing `args` are not used, so the
Picard coefficient update can share the argument list.

Constrained velocity components (`mesh.constrained_u`, `mesh.constrained_v`) and nodes with
neither stiffness nor drag (ice free) have identity rows, and constrained values enter the other rows as zero, so the
operator stays symmetric.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: triangle_ssa_operator!

julia> mesh = TriangleMesh(CPU(); x = [0.0, 3.0, 3.0, 0.0], y = [0.0, 0.0, 2.0, 2.0], elements = [1 1; 2 3; 3 4],
                           constrained = (falses(4), falses(4)));

julia> η̄H, β = ElementField(mesh), set!(NodeField(mesh), 2);

julia> p, q = fill!(VelocityVector(mesh), 3), VelocityVector(mesh);

julia> triangle_ssa_operator!(q, p, η̄H, β)

julia> interior(q.u) # without viscosity only the lumped drag remains, node area × 2 × 3
4-element Vector{Float64}:
 12.0
  6.0
 12.0
  6.0
```
"""
function triangle_ssa_operator!(q, p, η̄H, β, args...)
    mesh = η̄H.mesh
    launch!(architecture(mesh), mesh, length(mesh.x), _triangle_ssa_operator!, q.u, q.v, p.u, p.v, η̄H, β, mesh)
    return nothing
end

stress_balance_operator(::TriangleMesh) = triangle_ssa_operator!

# Viscous row of node n and its diagonal, gathered from the elements around the node.
@inline function viscous_row(n, u, v, η̄H, mesh)
    FT = eltype(mesh.area)
    qᵘ = qᵛ = dᵘ = dᵛ = zero(FT)
    @inbounds for position in mesh.node_element_offsets[n]:mesh.node_element_offsets[n+1]-1
        e = mesh.node_elements[position]
        i = mesh.node_local_index[position]
        bᵢ = mesh.∂φ∂x[i, e]
        cᵢ = mesh.∂φ∂y[i, e]
        η̄HA = η̄H.data[e] * mesh.area[e]
        dᵘ += η̄HA * (4bᵢ^2 + cᵢ^2)
        dᵛ += η̄HA * (4cᵢ^2 + bᵢ^2)
        for j in 1:3
            m = mesh.elements[j, e]
            bⱼ = mesh.∂φ∂x[j, e]
            cⱼ = mesh.∂φ∂y[j, e]
            uⱼ = ifelse(mesh.constrained_u[m], zero(FT), u.data[m])
            vⱼ = ifelse(mesh.constrained_v[m], zero(FT), v.data[m])
            qᵘ += η̄HA * ((4bⱼ * bᵢ + cⱼ * cᵢ) * uⱼ + (2cⱼ * bᵢ + bⱼ * cᵢ) * vⱼ)
            qᵛ += η̄HA * ((2bⱼ * cᵢ + cⱼ * bᵢ) * uⱼ + (4cⱼ * cᵢ + bⱼ * bᵢ) * vⱼ)
        end
    end
    return qᵘ, qᵛ, dᵘ, dᵛ
end

@kernel function _triangle_ssa_operator!(qu, qv, u, v, η̄H, β, mesh)
    n = @index(Global, Linear)
    qᵘ, qᵛ, dᵘ, dᵛ = viscous_row(n, u, v, η̄H, mesh)
    @inbounds begin
        Aβ = mesh.node_area[n] * β.data[n]
        identityᵘ = mesh.constrained_u[n] | iszero(dᵘ + Aβ)
        identityᵛ = mesh.constrained_v[n] | iszero(dᵛ + Aβ)
        qu.data[n] = ifelse(identityᵘ, u.data[n], qᵘ + Aβ * u.data[n])
        qv.data[n] = ifelse(identityᵛ, v.data[n], qᵛ + Aβ * v.data[n])
    end
end

function precondition!(z::VelocityVector{<:NodeField}, ::StressBalanceDiagonalPreconditioner, r::VelocityVector{<:NodeField},
                       η̄H::ElementField, β, args...)
    mesh = η̄H.mesh
    launch!(architecture(mesh), mesh, length(mesh.x), _triangle_diagonal_precondition!, z.u, z.v, r.u, r.v, η̄H, β, mesh)
    return nothing
end

@kernel function _triangle_diagonal_precondition!(zu, zv, ru, rv, η̄H, β, mesh)
    n = @index(Global, Linear)
    _, _, dᵘ, dᵛ = viscous_row(n, ru, rv, η̄H, mesh)
    @inbounds begin
        Aβ = mesh.node_area[n] * β.data[n]
        dᵘ += Aβ
        dᵛ += Aβ
        dᵘ = ifelse(mesh.constrained_u[n] | iszero(dᵘ), one(dᵘ), dᵘ)
        dᵛ = ifelse(mesh.constrained_v[n] | iszero(dᵛ), one(dᵛ), dᵛ)
        zu.data[n] = ru.data[n] / dᵘ
        zv.data[n] = rv.data[n] / dᵛ
    end
end
