"""
$(TYPEDEF)

Weertman sliding law `τ_b = C |u_b|^(m-1) u_b`, evaluated with the basal velocity `u_b`.
The speed is regularized, `|u_b|² + u₀²`, so the drag coefficient stays finite at rest.

$(TYPEDFIELDS)
"""
struct WeertmanFriction{C, FT}
    "friction coefficient `C`, a `Field` at `(Center, Center, Nothing)`"
    coefficient :: C
    "sliding exponent `m`"
    exponent :: FT
    "speed regularization `u₀` (m s⁻¹)"
    velocity_regularization :: FT
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
function WeertmanFriction(grid; coefficient, exponent = 1/3, velocity_regularization = 1e-8)
    FT = eltype(grid)
    return WeertmanFriction(parameter_field(grid, coefficient), convert(FT, exponent), convert(FT, velocity_regularization))
end

"""
$(TYPEDEF)

Regularized Coulomb sliding law (Schoof 2005, MISMIP+),
`τ_b = C |u_b|^m α² N / (C^(1/m) |u_b| + (α² N)^(1/m))^m · u_b / |u_b|`, which tends to
Weertman `C |u_b|^m` for slow sliding and to Coulomb `α² N` for fast sliding. `N` is the
effective pressure.

$(TYPEDFIELDS)
"""
struct RegularizedCoulombFriction{C, FT}
    "friction coefficient `C`, a `Field` at `(Center, Center, Nothing)`"
    coefficient :: C
    "sliding exponent `m`"
    exponent :: FT
    "Coulomb coefficient `α²`"
    coulomb_coefficient :: FT
    "speed regularization `u₀` (m s⁻¹)"
    velocity_regularization :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `RegularizedCoulombFriction` on `grid`; `coefficient` and `coulomb_coefficient` are required.
"""
function RegularizedCoulombFriction(grid; coefficient, coulomb_coefficient, exponent = 1/3, velocity_regularization = 1e-8)
    FT = eltype(grid)
    return RegularizedCoulombFriction(parameter_field(grid, coefficient),
                                      convert(FT, exponent),
                                      convert(FT, coulomb_coefficient),
                                      convert(FT, velocity_regularization))
end

Base.summary(f::WeertmanFriction) = string("WeertmanFriction(exponent = ", f.exponent, ")")
Base.summary(f::RegularizedCoulombFriction) = string("RegularizedCoulombFriction(exponent = ", f.exponent, ")")

Adapt.adapt_structure(to, f::WeertmanFriction) =
    WeertmanFriction(adapt(to, f.coefficient), f.exponent, f.velocity_regularization)

Adapt.adapt_structure(to, f::RegularizedCoulombFriction) =
    RegularizedCoulombFriction(adapt(to, f.coefficient), f.exponent, f.coulomb_coefficient, f.velocity_regularization)

# Drag coefficient β = τ_b / |u_b| from the speed `U`, friction coefficient `C` and effective pressure `N`.
@inline basal_drag(U, C, N, friction::WeertmanFriction) = C * U^(friction.exponent - 1)

@inline function basal_drag(U, C, N, friction::RegularizedCoulombFriction)
    m = friction.exponent
    τᶜ = friction.coulomb_coefficient * N
    return C * U^(m - 1) * τᶜ / (C^(1 / m) * U + τᶜ^(1 / m))^m
end

@inline x_speed²(i, j, k, grid, u, v) = @inbounds u[i, j, k]^2 + ℑxyᶠᶜᵃ(i, j, k, grid, v)^2
@inline y_speed²(i, j, k, grid, u, v) = @inbounds v[i, j, k]^2 + ℑxyᶜᶠᵃ(i, j, k, grid, u)^2

"""
$(TYPEDSIGNATURES)

Compute the basal drag coefficients `βᵘ`, `βᵛ` at the velocity points from the basal velocity
`(u, v)`, scaled by the grounded fractions `φᵘ`, `φᵛ` at the velocity points. `N` is the effective
pressure, used only by `RegularizedCoulombFriction`. `u`, `v`, `N` and the friction coefficient need filled halos.
With `friction = nothing` the drag coefficients are left unchanged.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_basal_drag!

julia> using Oceananigans.BoundaryConditions: fill_halo_regions!

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> u, v = XFaceField(grid), YFaceField(grid);

julia> set!(u, 1); set!(v, 0);

julia> βᵘ, βᵛ = XFaceField(grid), YFaceField(grid);

julia> φᵘ, φᵛ = XFaceField(grid), YFaceField(grid); set!(φᵘ, 1); set!(φᵛ, 1);

julia> friction = WeertmanFriction(grid; coefficient = 2, velocity_regularization = 0);

julia> compute_basal_drag!(βᵘ, βᵛ, friction, u, v, φᵘ, φᵛ, nothing)

julia> βᵘ[2, 1, 1]
2.0
```
"""
function compute_basal_drag!(βᵘ, βᵛ, friction, u, v, φᵘ, φᵛ, N)
    grid = u.grid
    arch = architecture(grid)
    launch!(arch, grid, size(βᵘ), _compute_x_basal_drag!, βᵘ, u, v, φᵘ, N, friction, grid)
    launch!(arch, grid, size(βᵛ), _compute_y_basal_drag!, βᵛ, u, v, φᵛ, N, friction, grid)
    return nothing
end

compute_basal_drag!(βᵘ, βᵛ, ::Nothing, args...) = nothing

@inline effective_pressureᶠᶜᶜ(i, j, k, grid, N) = ℑxᶠᵃᵃ(i, j, k, grid, N)
@inline effective_pressureᶜᶠᶜ(i, j, k, grid, N) = ℑyᵃᶠᵃ(i, j, k, grid, N)
@inline effective_pressureᶠᶜᶜ(i, j, k, grid, ::Nothing) = zero(grid)
@inline effective_pressureᶜᶠᶜ(i, j, k, grid, ::Nothing) = zero(grid)

@kernel function _compute_x_basal_drag!(βᵘ, u, v, φᵘ, N, friction, grid)
    i, j, k = @index(Global, NTuple)
    U = sqrt(x_speed²(i, j, k, grid, u, v) + friction.velocity_regularization^2)
    Cᶠ = ℑxᶠᵃᵃ(i, j, k, grid, friction.coefficient)
    Nᶠ = effective_pressureᶠᶜᶜ(i, j, k, grid, N)
    β = @inbounds φᵘ[i, j, k] * basal_drag(U, Cᶠ, Nᶠ, friction)
    wall = peripheral_node(i, j, k, grid, Face(), Center(), Center())
    @inbounds βᵘ[i, j, k] = ifelse(wall, zero(grid), β)
end

@kernel function _compute_y_basal_drag!(βᵛ, u, v, φᵛ, N, friction, grid)
    i, j, k = @index(Global, NTuple)
    U = sqrt(y_speed²(i, j, k, grid, u, v) + friction.velocity_regularization^2)
    Cᶠ = ℑyᵃᶠᵃ(i, j, k, grid, friction.coefficient)
    Nᶠ = effective_pressureᶜᶠᶜ(i, j, k, grid, N)
    β = @inbounds φᵛ[i, j, k] * basal_drag(U, Cᶠ, Nᶠ, friction)
    wall = peripheral_node(i, j, k, grid, Center(), Face(), Center())
    @inbounds βᵛ[i, j, k] = ifelse(wall, zero(grid), β)
end
