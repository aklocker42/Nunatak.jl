"""
$(TYPEDSIGNATURES)

Compute the area-integrated driving stress `τ = -Az ρᵢ g H ∇s` at the velocity points,
the right-hand side of `ssa_operator!`. Wall faces, and faces that are ice free in the mask `μ` when
it is given, get zero. `H`, `s` and `μ` need filled halos.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_driving_stress!

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> H, s = CenterField(grid), CenterField(grid);

julia> set!(H, 1); set!(s, (x, y) -> x);

julia> τ = VelocityVector(grid);

julia> compute_driving_stress!(τ, H, s, 1, 1)

julia> τ.u[2, 1, 1]
-0.0625
```
"""
function compute_driving_stress!(τ, H, s, ice_density, gravitational_acceleration, μ = nothing)
    grid = H.grid
    FT = eltype(grid)
    ρg = convert(FT, ice_density * gravitational_acceleration)
    arch = architecture(grid)
    launch!(arch, grid, size(τ.u), _compute_x_driving_stress!, τ.u, H, s, μ, ρg, grid)
    launch!(arch, grid, size(τ.v), _compute_y_driving_stress!, τ.v, H, s, μ, ρg, grid)
    return nothing
end

@kernel function _compute_x_driving_stress!(τu, H, s, μ, ρg, grid)
    i, j, k = @index(Global, NTuple)
    wall = inactiveᶠᶜᶜ(i, j, k, grid, μ)
    τ = - Azᶠᶜᶜ(i, j, k, grid) * ρg * ℑxᶠᵃᵃ(i, j, k, grid, H) * ∂xᶠᶜᶜ(i, j, k, grid, s)
    @inbounds τu[i, j, k] = ifelse(wall, zero(grid), τ)
end

@kernel function _compute_y_driving_stress!(τv, H, s, μ, ρg, grid)
    i, j, k = @index(Global, NTuple)
    wall = inactiveᶜᶠᶜ(i, j, k, grid, μ)
    τ = - Azᶜᶠᶜ(i, j, k, grid) * ρg * ℑyᵃᶠᵃ(i, j, k, grid, H) * ∂yᶜᶠᶜ(i, j, k, grid, s)
    @inbounds τv[i, j, k] = ifelse(wall, zero(grid), τ)
end
