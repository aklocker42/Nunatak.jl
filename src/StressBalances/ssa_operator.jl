"""
$(TYPEDSIGNATURES)

Apply the linearized depth-integrated stress-balance operator to the velocity `p` and store
the result in `q`:

`q = - ∇ ⋅ 𝐓 + β p`, with membrane stresses `Tˣˣ = 2 η̄H (2 ∂u/∂x + ∂v/∂y)`,
`Tʸʸ = 2 η̄H (2 ∂v/∂y + ∂u/∂x)` at cell centers and `Tˣʸ = η̄H (∂u/∂y + ∂v/∂x)` at cell corners. Each row is integrated over its
velocity cell, so the operator is symmetric positive definite on any `RectilinearGrid`
and the right-hand side must be area-integrated as well. SSA passes `βᵘ, βᵛ = β`;
DIVA passes `β / (1 + β F₂)`. Of the trailing `args` only the wall condition, `FreeSlipWalls()`
or `NoSlipWalls()`, is used and it is required, so the Picard coefficient update can share the argument list.

Velocities on `Bounded` walls are impenetrable: their rows are the identity and they are
excluded from the strain rates, so the wall values of `p` do not matter. The tangential condition
sets the shear at wall corners. Velocity points with no stiffness on either side (`η̄H = 0`, ice free) are treated the
same way, so the operator stays symmetric as the ice extent changes.
The halos of `p`, `η̄H` and `η̄Hᶠᶠ` must be filled.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Oceananigans.BoundaryConditions: fill_halo_regions!

julia> using Nunatak: ssa_operator!

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> η̄H = Field{Center, Center, Nothing}(grid); η̄Hᶠᶠ = Field{Face, Face, Nothing}(grid);

julia> set!(η̄H, 1); set!(η̄Hᶠᶠ, 1); fill_halo_regions!((η̄H, η̄Hᶠᶠ));

julia> βᵘ = Field{Face, Center, Nothing}(grid); βᵛ = Field{Center, Face, Nothing}(grid);

julia> set!(βᵘ, 2); set!(βᵛ, 2);

julia> p, q = VelocityVector(grid), VelocityVector(grid);

julia> set!(p.u, 3); fill_halo_regions!(p);

julia> ssa_operator!(q, p, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, FreeSlipWalls())

julia> interior(q.u)[1, 1, 1] # rigid translation feels only the drag, 2 × 3 × (1/4)²
0.375
```
"""
function ssa_operator!(q, p, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, args...)
    grid = η̄H.grid
    walls = side_walls(args...)
    launch!(architecture(grid), grid, :xyz, _ssa_operator!, q.u, q.v, p.u, p.v, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid)
    return nothing
end

const SideWalls = Union{FreeSlipWalls, NoSlipWalls}

side_walls() = throw(ArgumentError("the coefficients must include the side walls, FreeSlipWalls() or NoSlipWalls()"))
side_walls(walls::SideWalls, args...) = walls
side_walls(coefficient, args...) = side_walls(args...)

@inline normal_stress_xx(i, j, k, grid, u, v, η̄H) = @inbounds 2 * η̄H[i, j, k] * (2 * strain_rate_xx(i, j, k, grid, u, η̄H) + strain_rate_yy(i, j, k, grid, v, η̄H))
@inline normal_stress_yy(i, j, k, grid, u, v, η̄H) = @inbounds 2 * η̄H[i, j, k] * (2 * strain_rate_yy(i, j, k, grid, v, η̄H) + strain_rate_xx(i, j, k, grid, u, η̄H))
@inline shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ, walls, η̄H) = @inbounds 2 * η̄Hᶠᶠ[i, j, k] * strain_rate_xy(i, j, k, grid, u, v, walls, η̄H)

@inline Δy_normal_stress_xx(i, j, k, grid, u, v, η̄H) = Δyᶜᶜᶜ(i, j, k, grid) * normal_stress_xx(i, j, k, grid, u, v, η̄H)
@inline Δx_normal_stress_yy(i, j, k, grid, u, v, η̄H) = Δxᶜᶜᶜ(i, j, k, grid) * normal_stress_yy(i, j, k, grid, u, v, η̄H)
@inline Δx_shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ, walls, η̄H) = Δxᶠᶠᶜ(i, j, k, grid) * shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ, walls, η̄H)
@inline Δy_shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ, walls, η̄H) = Δyᶠᶠᶜ(i, j, k, grid) * shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ, walls, η̄H)

@inline function x_stress_balance(i, j, k, grid, u, v, η̄H, η̄Hᶠᶠ, βᵘ, walls)
    divergence = δxᶠᶜᶜ(i, j, k, grid, Δy_normal_stress_xx, u, v, η̄H) +
                 δyᶠᶜᶜ(i, j, k, grid, Δx_shear_stress, u, v, η̄Hᶠᶠ, walls, η̄H)
    return @inbounds Azᶠᶜᶜ(i, j, k, grid) * βᵘ[i, j, k] * u[i, j, k] - divergence
end

@inline function y_stress_balance(i, j, k, grid, u, v, η̄H, η̄Hᶠᶠ, βᵛ, walls)
    divergence = δyᶜᶠᶜ(i, j, k, grid, Δx_normal_stress_yy, u, v, η̄H) +
                 δxᶜᶠᶜ(i, j, k, grid, Δy_shear_stress, u, v, η̄Hᶠᶠ, walls, η̄H)
    return @inbounds Azᶜᶠᶜ(i, j, k, grid) * βᵛ[i, j, k] * v[i, j, k] - divergence
end

@kernel function _ssa_operator!(qu, qv, u, v, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid)
    i, j, k = @index(Global, NTuple)
    wallᵘ = inactiveᶠᶜᶜ(i, j, k, grid, η̄H)
    wallᵛ = inactiveᶜᶠᶜ(i, j, k, grid, η̄H)
    @inbounds begin
        qu[i, j, k] = ifelse(wallᵘ, u[i, j, k], x_stress_balance(i, j, k, grid, u, v, η̄H, η̄Hᶠᶠ, βᵘ, walls))
        qv[i, j, k] = ifelse(wallᵛ, v[i, j, k], y_stress_balance(i, j, k, grid, u, v, η̄H, η̄Hᶠᶠ, βᵛ, walls))
    end
end
