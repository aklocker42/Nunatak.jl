"""
$(TYPEDSIGNATURES)

Compute the area-integrated driving stress `τ = -Az ρᵢ g H ∇s` at the velocity points,
the right-hand side of `ssa_operator!`. Wall faces, and faces that are ice free in the mask `μ` when
it is given, get zero. At calving fronts, faces with ice on one side only, `τ` is the net front force
`Δ ⋅ g (ρᵢ H² - ρw D²) / 2` of the ice cell, with `D` its depth below sea level.
`H`, `s` and `μ` need filled halos.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_driving_stress!

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> H, s = CenterField(grid), CenterField(grid);

julia> set!(H, 1); set!(s, (x, y) -> x);

julia> τ = VelocityVector(grid);

julia> compute_driving_stress!(τ, H, s, 1, 1, 1)

julia> τ.u[2, 1, 1]
-0.0625
```
"""
function compute_driving_stress!(τ, H, s, ice_density, ocean_density, gravitational_acceleration, μ = nothing)
    grid = H.grid
    FT = eltype(grid)
    ρᵢ = convert(FT, ice_density)
    ρw = convert(FT, ocean_density)
    g = convert(FT, gravitational_acceleration)
    arch = architecture(grid)
    launch!(arch, grid, size(τ.u), _compute_x_driving_stress!, τ.u, H, s, μ, ρᵢ, ρw, g, grid)
    launch!(arch, grid, size(τ.v), _compute_y_driving_stress!, τ.v, H, s, μ, ρᵢ, ρw, g, grid)
    return nothing
end

# Depth-integrated ice pressure minus ocean pressure on the submerged part D of a front cell (sea level at z = 0).
@inline function front_force(i, j, k, grid, H, s, ρᵢ, ρw, g)
    Hᵢ = @inbounds H[i, j, k]
    D = clamp(Hᵢ - @inbounds(s[i, j, k]), zero(grid), Hᵢ)
    return g * (ρᵢ * Hᵢ^2 - ρw * D^2) / 2
end

@inline has_ice(i, j, k, μ) = @inbounds μ[i, j, k] > 0

@inline function x_front_stress(i, j, k, grid, H, s, μ, ρᵢ, ρw, g)
    west_ice = has_ice(i-1, j, k, μ) & !has_ice(i, j, k, μ)
    east_ice = !has_ice(i-1, j, k, μ) & has_ice(i, j, k, μ)
    F = ifelse(west_ice, front_force(i-1, j, k, grid, H, s, ρᵢ, ρw, g), zero(grid)) -
        ifelse(east_ice, front_force(i, j, k, grid, H, s, ρᵢ, ρw, g), zero(grid))
    return west_ice | east_ice, Δyᶠᶜᶜ(i, j, k, grid) * F
end

@inline function y_front_stress(i, j, k, grid, H, s, μ, ρᵢ, ρw, g)
    south_ice = has_ice(i, j-1, k, μ) & !has_ice(i, j, k, μ)
    north_ice = !has_ice(i, j-1, k, μ) & has_ice(i, j, k, μ)
    F = ifelse(south_ice, front_force(i, j-1, k, grid, H, s, ρᵢ, ρw, g), zero(grid)) -
        ifelse(north_ice, front_force(i, j, k, grid, H, s, ρᵢ, ρw, g), zero(grid))
    return south_ice | north_ice, Δxᶜᶠᶜ(i, j, k, grid) * F
end

@inline x_front_stress(i, j, k, grid, H, s, ::Nothing, args...) = false, zero(grid)
@inline y_front_stress(i, j, k, grid, H, s, ::Nothing, args...) = false, zero(grid)

@kernel function _compute_x_driving_stress!(τu, H, s, μ, ρᵢ, ρw, g, grid)
    i, j, k = @index(Global, NTuple)
    wall = inactiveᶠᶜᶜ(i, j, k, grid, μ)
    front, τᶠ = x_front_stress(i, j, k, grid, H, s, μ, ρᵢ, ρw, g)
    τ = - Azᶠᶜᶜ(i, j, k, grid) * ρᵢ * g * ℑxᶠᵃᵃ(i, j, k, grid, H) * ∂xᶠᶜᶜ(i, j, k, grid, s)
    @inbounds τu[i, j, k] = ifelse(wall, zero(grid), ifelse(front, τᶠ, τ))
end

@kernel function _compute_y_driving_stress!(τv, H, s, μ, ρᵢ, ρw, g, grid)
    i, j, k = @index(Global, NTuple)
    wall = inactiveᶜᶠᶜ(i, j, k, grid, μ)
    front, τᶠ = y_front_stress(i, j, k, grid, H, s, μ, ρᵢ, ρw, g)
    τ = - Azᶜᶠᶜ(i, j, k, grid) * ρᵢ * g * ℑyᵃᶠᵃ(i, j, k, grid, H) * ∂yᶜᶠᶜ(i, j, k, grid, s)
    @inbounds τv[i, j, k] = ifelse(wall, zero(grid), ifelse(front, τᶠ, τ))
end
