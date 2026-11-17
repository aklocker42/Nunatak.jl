"""
$(TYPEDSIGNATURES)

Compute the depth-integrated Glen viscosity `η̄H` of each element of a `TriangleMesh` from the nodal velocities
`u`, `v` and thickness `H`. The P1 strain rates are uniform in an element, `ε̇ₑ² = ε̇ₓₓ² + ε̇ᵧᵧ² + ε̇ₓₓ ε̇ᵧᵧ + ε̇ₓᵧ² + ε̇₀²`,
and `η̄H = η(ε̇ₑ², Ā) H̄` with `Ā` and `H̄` the element means of the nodal rate factor and thickness.
Constrained velocity components enter as zero.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_element_viscosity!

julia> mesh = TriangleMesh(CPU(); x = [0.0, 1.0, 1.0, 0.0], y = [0.0, 0.0, 1.0, 1.0], elements = [1 1; 2 3; 3 4]);

julia> rheology = GlenFlowLaw(mesh; rate_factor = 1, exponent = 1);

julia> η̄H, H = ElementField(mesh), set!(NodeField(mesh), 4);

julia> u = VelocityVector(mesh);

julia> compute_element_viscosity!(η̄H, rheology, u.u, u.v, H)

julia> interior(η̄H) # ½ A⁻¹ H for a linear rheology
2-element Vector{Float64}:
 2.0
 2.0
```
"""
function compute_element_viscosity!(η̄H, rheology::GlenFlowLaw, u, v, H)
    mesh = H.mesh
    n = rheology.exponent
    ε̇₀ = rheology.strain_rate_regularization
    launch!(architecture(mesh), mesh, length(η̄H), _compute_element_viscosity!, η̄H, u, v, H, rheology.rate_factor, n, ε̇₀, mesh)
    return nothing
end

@kernel function _compute_element_viscosity!(η̄H, u, v, H, A, n, ε̇₀, mesh)
    e = @index(Global, Linear)
    FT = eltype(mesh.area)
    ε̇xx = ε̇yy = ε̇xy = Ā = H̄ = zero(FT)
    @inbounds for i in 1:3
        m = mesh.elements[i, e]
        uᵢ = ifelse(mesh.constrained_u[m], zero(FT), u.data[m])
        vᵢ = ifelse(mesh.constrained_v[m], zero(FT), v.data[m])
        ε̇xx += mesh.∂φ∂x[i, e] * uᵢ
        ε̇yy += mesh.∂φ∂y[i, e] * vᵢ
        ε̇xy += (mesh.∂φ∂y[i, e] * uᵢ + mesh.∂φ∂x[i, e] * vᵢ) / 2
        Ā += A.data[m] / 3
        H̄ += H.data[m] / 3
    end
    ε̇ₑ² = ε̇xx^2 + ε̇yy^2 + ε̇xx * ε̇yy + ε̇xy^2 + ε̇₀^2
    @inbounds η̄H.data[e] = glen_viscosity(ε̇ₑ², Ā, n) * H̄
end

"""
$(TYPEDSIGNATURES)

Compute the nodal basal drag coefficient `β = φ βᶠ(|u|, C, N)` on a `TriangleMesh` from the nodal velocity `(u, v)`,
grounded flag `φ` and effective pressure `N`, with `βᶠ` the drag of `friction`. The drag is lumped to the nodes, as on the C-grid;
a consistent mass matrix would differ at second order in the element size. With `friction = nothing` `β` is left unchanged.
"""
function compute_nodal_basal_drag!(β, friction, u, v, φ, N)
    mesh = β.mesh
    launch!(architecture(mesh), mesh, length(β), _compute_nodal_basal_drag!, β, u, v, φ, N, friction)
    return nothing
end

compute_nodal_basal_drag!(β, ::Nothing, args...) = nothing

@kernel function _compute_nodal_basal_drag!(β, u, v, φ, N, friction)
    n = @index(Global, Linear)
    @inbounds begin
        U = sqrt(u.data[n]^2 + v.data[n]^2 + friction.velocity_regularization^2)
        β.data[n] = φ.data[n] * basal_drag(U, friction.coefficient.data[n], N.data[n], friction)
    end
end

"""
$(TYPEDSIGNATURES)

Freeze the stress-balance coefficients on a `TriangleMesh` at the velocity `x`: element viscosity `η̄H` and nodal drag `β`.
The argument list after `x` is the one `triangle_ssa_operator!` takes, so this is the `update_coefficients!` of `picard_solve!`.
"""
function update_triangle_stress_balance_coefficients!(x, η̄H, β, H, rheology, friction, φ, N)
    compute_element_viscosity!(η̄H, rheology, x.u, x.v, H)
    compute_nodal_basal_drag!(β, friction, x.u, x.v, φ, N)
    return nothing
end

overwritten_coefficients(::typeof(update_triangle_stress_balance_coefficients!), η̄H, β, H, rheology, friction, args...) = (η̄H, β)
overwritten_coefficients(::typeof(update_triangle_stress_balance_coefficients!), η̄H, β, H, rheology, ::Nothing, args...) = (η̄H,)

"""
$(TYPEDSIGNATURES)

Compute the driving stress on a `TriangleMesh`, `τᵢ = -ρᵢ g ∫ H ∇s φᵢ dA`, integrated exactly for P1 `H` and `s`,
plus the ice-front force `∫ g (ρᵢ H² - ρw D²) / 2 n̂ φᵢ dl` on `mesh.front_segments`, with `D` the depth below sea level
and `n̂` the outward normal, by two-point Gauss quadrature along each segment. Constrained components get zero.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_driving_stress!

julia> mesh = TriangleMesh(CPU(); x = [0.0, 3.0, 3.0, 0.0], y = [0.0, 0.0, 2.0, 2.0], elements = [1 1; 2 3; 3 4],
                           constrained = (falses(4), falses(4)));

julia> H, s = set!(NodeField(mesh), 1), set!(NodeField(mesh), (x, y) -> 3y);

julia> τ = VelocityVector(mesh);

julia> compute_driving_stress!(τ, H, s, 1, 1, 1)

julia> interior(τ.v) # -∂s/∂y × node area
4-element Vector{Float64}:
 -6.0
 -3.0
 -6.0
 -3.0
```
"""
function compute_driving_stress!(τ, H::NodeField, s, ice_density, ocean_density, gravitational_acceleration)
    mesh = H.mesh
    FT = eltype(mesh)
    ρᵢ = convert(FT, ice_density)
    ρw = convert(FT, ocean_density)
    g = convert(FT, gravitational_acceleration)
    arch = architecture(mesh)
    launch!(arch, mesh, length(H), _compute_triangle_driving_stress!, τ.u, τ.v, H, s, ρᵢ, g, mesh)
    # Neighbouring front segments share nodes, so one work item adds all of them.
    launch!(arch, mesh, min(1, size(mesh.front_segments, 2)), _add_front_stress!, τ.u, τ.v, H, s, ρᵢ, ρw, g, mesh)
    return nothing
end

@kernel function _compute_triangle_driving_stress!(τu, τv, H, s, ρᵢ, g, mesh)
    n = @index(Global, Linear)
    FT = eltype(mesh.area)
    τᵘ = τᵛ = zero(FT)
    @inbounds for position in mesh.node_element_offsets[n]:mesh.node_element_offsets[n+1]-1
        e = mesh.node_elements[position]
        ∂s∂x = ∂s∂y = ΣH = zero(FT)
        for j in 1:3
            m = mesh.elements[j, e]
            ∂s∂x += mesh.∂φ∂x[j, e] * s.data[m]
            ∂s∂y += mesh.∂φ∂y[j, e] * s.data[m]
            ΣH += H.data[m]
        end
        ∫Hφ = mesh.area[e] * (H.data[n] + ΣH) / 12
        τᵘ -= ρᵢ * g * ∂s∂x * ∫Hφ
        τᵛ -= ρᵢ * g * ∂s∂y * ∫Hφ
    end
    @inbounds begin
        τu.data[n] = ifelse(mesh.constrained_u[n], zero(FT), τᵘ)
        τv.data[n] = ifelse(mesh.constrained_v[n], zero(FT), τᵛ)
    end
end

@inline function triangle_front_force(H, s, ρᵢ, ρw, g)
    D = clamp(H - s, zero(H), H)
    return g * (ρᵢ * H^2 - ρw * D^2) / 2
end

@kernel function _add_front_stress!(τu, τv, H, s, ρᵢ, ρw, g, mesh)
    FT = eltype(mesh.area)
    ξ₁ = (1 - 1 / sqrt(convert(FT, 3))) / 2
    @inbounds for f in axes(mesh.front_segments, 2)
        n₁ = mesh.front_segments[1, f]
        n₂ = mesh.front_segments[2, f]
        F₁ = F₂ = zero(FT)
        for ξ in (ξ₁, 1 - ξ₁)
            Hξ = (1 - ξ) * H.data[n₁] + ξ * H.data[n₂]
            sξ = (1 - ξ) * s.data[n₁] + ξ * s.data[n₂]
            F = triangle_front_force(Hξ, sξ, ρᵢ, ρw, g) * mesh.front_lengths[f] / 2
            F₁ += (1 - ξ) * F
            F₂ += ξ * F
        end
        nx = mesh.front_normals[1, f]
        ny = mesh.front_normals[2, f]
        τu.data[n₁] += ifelse(mesh.constrained_u[n₁], zero(FT), nx * F₁)
        τv.data[n₁] += ifelse(mesh.constrained_v[n₁], zero(FT), ny * F₁)
        τu.data[n₂] += ifelse(mesh.constrained_u[n₂], zero(FT), nx * F₂)
        τv.data[n₂] += ifelse(mesh.constrained_v[n₂], zero(FT), ny * F₂)
    end
end

"""
$(TYPEDSIGNATURES)

Return the nodal geometry fields of a `TriangleMesh`: thickness `H`, surface `s`, bed `b`, grounded flag `φ` and effective
pressure `N`, all zero.
"""
ice_sheet_geometry(mesh::TriangleMesh) =
    (; H = NodeField(mesh), s = NodeField(mesh), b = NodeField(mesh), φ = NodeField(mesh), N = NodeField(mesh))

"""
$(TYPEDSIGNATURES)

Return the stress-balance work fields of a `TriangleMesh`: element viscosity `η̄H`, nodal drag `β` and driving stress `τ`.
"""
ice_sheet_auxiliary_fields(mesh::TriangleMesh) = (; η̄H = ElementField(mesh), β = NodeField(mesh), τ = VelocityVector(mesh))

"""
$(TYPEDSIGNATURES)

Update the nodal surface `s = max(b + H, (1 - ρᵢ / ρw) H)`, grounded flag `φ`, one where `H + ρw b / ρᵢ > 0`, and effective
pressure `N = max(ρᵢ g H + ρw g min(b, 0), 0)` of a `TriangleMesh` geometry from its thickness `H` and bed `b`.
"""
function update_geometry!(geometry, constants)
    (; H, s, b, φ, N) = geometry
    mesh = H.mesh
    ρᵢ = constants.ice_density
    ρw = constants.ocean_density
    g = constants.gravitational_acceleration
    launch!(architecture(mesh), mesh, length(H), _update_triangle_geometry!, s, φ, N, H, b, ρᵢ, ρw, g)
    return nothing
end

@kernel function _update_triangle_geometry!(s, φ, N, H, b, ρᵢ, ρw, g)
    n = @index(Global, Linear)
    @inbounds begin
        Hₙ = H.data[n]
        bₙ = b.data[n]
        s.data[n] = max(bₙ + Hₙ, (1 - ρᵢ / ρw) * Hₙ)
        φ.data[n] = ifelse(Hₙ + ρw / ρᵢ * bₙ > 0, one(Hₙ), zero(Hₙ))
        N.data[n] = max(ρᵢ * g * Hₙ + ρw * g * min(bₙ, zero(bₙ)), zero(Hₙ))
    end
end

"""
$(TYPEDSIGNATURES)

Solve the shallow-shelf stress balance on a `TriangleMesh` for the nodal velocity `x`, starting from its current value:
update the `geometry` from `H` and `b`, compute the driving stress, then run the Picard iterations of `solver`.
`geometry` and `auxiliary` come from `ice_sheet_geometry(mesh)` and `ice_sheet_auxiliary_fields(mesh)`; `friction` may be `nothing`.
Returns `x`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: ice_sheet_geometry, ice_sheet_auxiliary_fields

julia> L = 1e4; x = [0, L, L, 0, 2L, 2L]; y = [0, 0, L, L, 0, L];

julia> mesh = TriangleMesh(CPU(); x, y, elements = [1 1 2 2; 2 3 5 6; 3 4 6 3], front_segments = [5; 6;;],
                           constrained = ((x, y) -> x == 0, (x, y) -> y == 0 || y == L));

julia> geometry, auxiliary = ice_sheet_geometry(mesh), ice_sheet_auxiliary_fields(mesh);

julia> set!(geometry.H, 500); set!(geometry.b, -1000);

julia> rheology = GlenFlowLaw(mesh; rate_factor = 1e-15, exponent = 1);

julia> velocity = VelocityVector(mesh);

julia> solve_stress_balance!(velocity, StressBalanceSolver(mesh), geometry, auxiliary, rheology, nothing, IceSheetConstants(mesh));

julia> free = [2, 3, 5, 6];

julia> round.(interior(velocity.u)[free] ./ x[free] .* 1e10; digits = 6) # uniform strain rate of a free shelf (10⁻¹⁰ s⁻¹)
4-element Vector{Float64}:
 1.20454
 1.20454
 1.20454
 1.20454
```
"""
function solve_stress_balance!(x, solver::StressBalanceSolver, geometry, auxiliary, rheology, friction, constants)
    (; H, s, φ, N) = geometry
    (; η̄H, β, τ) = auxiliary
    update_geometry!(geometry, constants)
    compute_driving_stress!(τ, H, s, constants.ice_density, constants.ocean_density, constants.gravitational_acceleration)
    coefficients = (η̄H, β, H, rheology, friction, φ, N)
    solver.iteration = picard_solve!(x, solver.picard, solver.linear_solver, update_triangle_stress_balance_coefficients!, τ, coefficients...)
    return x
end
