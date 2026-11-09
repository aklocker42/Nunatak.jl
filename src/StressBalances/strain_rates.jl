# Wall velocities, and ice-free ones when the stiffness `η̄H` is given, are not unknowns, so strain rates see them as zero.
@inline inactiveᶠᶜᶜ(i, j, k, grid, η̄H = nothing) = peripheral_node(i, j, k, grid, Face(), Center(), Center()) | ice_freeᶠᶜᶜ(i, j, k, grid, η̄H)
@inline inactiveᶜᶠᶜ(i, j, k, grid, η̄H = nothing) = peripheral_node(i, j, k, grid, Center(), Face(), Center()) | ice_freeᶜᶠᶜ(i, j, k, grid, η̄H)

@inline interior_u(i, j, k, grid, u, η̄H = nothing) = @inbounds ifelse(inactiveᶠᶜᶜ(i, j, k, grid, η̄H), zero(grid), u[i, j, k])
@inline interior_v(i, j, k, grid, v, η̄H = nothing) = @inbounds ifelse(inactiveᶜᶠᶜ(i, j, k, grid, η̄H), zero(grid), v[i, j, k])

@inline strain_rate_xx(i, j, k, grid, u, args...) = ∂xᶜᶜᶜ(i, j, k, grid, interior_u, u, args...)
@inline strain_rate_yy(i, j, k, grid, v, args...) = ∂yᶜᶜᶜ(i, j, k, grid, interior_v, v, args...)
# With a zero velocity behind the wall, a mirror ghost doubles the corner shear; free slip removes it.
@inline wall_shear_factor(::FreeSlipWalls, grid) = zero(grid)
@inline wall_shear_factor(::NoSlipWalls, grid) = 2 * one(grid)

@inline function strain_rate_xy(i, j, k, grid, u, v, walls, args...)
    ε̇xy = (∂yᶠᶠᶜ(i, j, k, grid, interior_u, u, args...) + ∂xᶠᶠᶜ(i, j, k, grid, interior_v, v, args...)) / 2
    return ifelse(peripheral_node(i, j, k, grid, Face(), Face(), Center()), wall_shear_factor(walls, grid) * ε̇xy, ε̇xy)
end

@inline strain_rate_xy²(i, j, k, grid, u, v, walls) = strain_rate_xy(i, j, k, grid, u, v, walls)^2

"""
$(TYPEDSIGNATURES)

Square of the membrane effective strain rate at cell centres,
`ε̇ₑ² = ε̇ₓₓ² + ε̇ᵧᵧ² + ε̇ₓₓ ε̇ᵧᵧ + ε̇ₓᵧ²`, with `ε̇ₓᵧ²` averaged from the cell corners under the wall condition `walls`.
"""
@inline function effective_strain_rate²(i, j, k, grid, u, v, walls = FreeSlipWalls())
    ε̇xx = strain_rate_xx(i, j, k, grid, u)
    ε̇yy = strain_rate_yy(i, j, k, grid, v)
    ε̇xy² = ℑxyᶜᶜᵃ(i, j, k, grid, strain_rate_xy², u, v, walls)
    return ε̇xx^2 + ε̇yy^2 + ε̇xx * ε̇yy + ε̇xy²
end
