"""
$(TYPEDSIGNATURES)

Apply the linearised depth-integrated stress-balance operator to the velocity `p` and store
the result in `q`,

```math
q = - ∇ ⋅ 𝐓 + β_\\mathrm{eff} \\, p ,
```

with membrane stresses ``T_{xx} = 2 \\bar{η} H (2 ∂_x u + ∂_y v)``,
``T_{yy} = 2 \\bar{η} H (2 ∂_y v + ∂_x u)`` at cell centres and
``T_{xy} = \\bar{η} H (∂_y u + ∂_x v)`` at cell corners. Each row is integrated over its
velocity cell, so the operator is symmetric positive definite on any `RectilinearGrid`
and the right-hand side must be area-integrated as well. SSA passes `βᵘ, βᵛ = β`;
DIVA passes `β / (1 + β F₂)`. Trailing `args` are ignored, so the Picard coefficient update
can share the argument list.

Velocities on `Bounded` walls are impenetrable: their rows are the identity and they are
excluded from the strain rates, so the wall values of `p` do not matter. Tangential walls are
free slip. The halos of `p`, `η̄H` and `η̄Hᶠᶠ` must be filled.

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

julia> ssa_operator!(q, p, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ)

julia> interior(q.u)[1, 1, 1] # rigid translation feels only the drag, 2 × 3 × (1/4)²
0.375
```
"""
function ssa_operator!(q, p, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, args...)
    grid = η̄H.grid
    launch!(architecture(grid), grid, :xyz, _ssa_operator!, q.u, q.v, p.u, p.v, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, grid)
    return nothing
end

@inline normal_stress_xx(i, j, k, grid, u, v, η̄H) = @inbounds 2 * η̄H[i, j, k] * (2 * strain_rate_xx(i, j, k, grid, u) + strain_rate_yy(i, j, k, grid, v))
@inline normal_stress_yy(i, j, k, grid, u, v, η̄H) = @inbounds 2 * η̄H[i, j, k] * (2 * strain_rate_yy(i, j, k, grid, v) + strain_rate_xx(i, j, k, grid, u))
@inline shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ)   = @inbounds 2 * η̄Hᶠᶠ[i, j, k] * strain_rate_xy(i, j, k, grid, u, v)

@inline Δy_normal_stress_xx(i, j, k, grid, u, v, η̄H) = Δyᶜᶜᶜ(i, j, k, grid) * normal_stress_xx(i, j, k, grid, u, v, η̄H)
@inline Δx_normal_stress_yy(i, j, k, grid, u, v, η̄H) = Δxᶜᶜᶜ(i, j, k, grid) * normal_stress_yy(i, j, k, grid, u, v, η̄H)
@inline Δx_shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ)   = Δxᶠᶠᶜ(i, j, k, grid) * shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ)
@inline Δy_shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ)   = Δyᶠᶠᶜ(i, j, k, grid) * shear_stress(i, j, k, grid, u, v, η̄Hᶠᶠ)

@inline function x_stress_balance(i, j, k, grid, u, v, η̄H, η̄Hᶠᶠ, βᵘ)
    divergence = δxᶠᶜᶜ(i, j, k, grid, Δy_normal_stress_xx, u, v, η̄H) +
                 δyᶠᶜᶜ(i, j, k, grid, Δx_shear_stress, u, v, η̄Hᶠᶠ)
    return @inbounds Azᶠᶜᶜ(i, j, k, grid) * βᵘ[i, j, k] * u[i, j, k] - divergence
end

@inline function y_stress_balance(i, j, k, grid, u, v, η̄H, η̄Hᶠᶠ, βᵛ)
    divergence = δyᶜᶠᶜ(i, j, k, grid, Δx_normal_stress_yy, u, v, η̄H) +
                 δxᶜᶠᶜ(i, j, k, grid, Δy_shear_stress, u, v, η̄Hᶠᶠ)
    return @inbounds Azᶜᶠᶜ(i, j, k, grid) * βᵛ[i, j, k] * v[i, j, k] - divergence
end

@kernel function _ssa_operator!(qu, qv, u, v, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, grid)
    i, j, k = @index(Global, NTuple)
    wallᵘ = peripheral_node(i, j, k, grid, Face(), Center(), Center())
    wallᵛ = peripheral_node(i, j, k, grid, Center(), Face(), Center())
    @inbounds begin
        qu[i, j, k] = ifelse(wallᵘ, u[i, j, k], x_stress_balance(i, j, k, grid, u, v, η̄H, η̄Hᶠᶠ, βᵘ))
        qv[i, j, k] = ifelse(wallᵛ, v[i, j, k], y_stress_balance(i, j, k, grid, u, v, η̄H, η̄Hᶠᶠ, βᵛ))
    end
end
