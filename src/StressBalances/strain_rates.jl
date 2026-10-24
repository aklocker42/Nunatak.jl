# Wall velocities are boundary values, not unknowns, so strain rates see them as zero.
@inline interior_u(i, j, k, grid, u) = @inbounds ifelse(peripheral_node(i, j, k, grid, Face(), Center(), Center()), zero(grid), u[i, j, k])
@inline interior_v(i, j, k, grid, v) = @inbounds ifelse(peripheral_node(i, j, k, grid, Center(), Face(), Center()), zero(grid), v[i, j, k])

@inline strain_rate_xx(i, j, k, grid, u) = ∂xᶜᶜᶜ(i, j, k, grid, interior_u, u)
@inline strain_rate_yy(i, j, k, grid, v) = ∂yᶜᶜᶜ(i, j, k, grid, interior_v, v)
@inline strain_rate_xy(i, j, k, grid, u, v) = (∂yᶠᶠᶜ(i, j, k, grid, interior_u, u) + ∂xᶠᶠᶜ(i, j, k, grid, interior_v, v)) / 2

@inline strain_rate_xy²(i, j, k, grid, u, v) = strain_rate_xy(i, j, k, grid, u, v)^2

"""
$(TYPEDSIGNATURES)

Square of the membrane effective strain rate at cell centres,
`ε̇ₑ² = ε̇ₓₓ² + ε̇ᵧᵧ² + ε̇ₓₓ ε̇ᵧᵧ + ε̇ₓᵧ²`, with `ε̇ₓᵧ²` averaged from the cell corners.
"""
@inline function effective_strain_rate²(i, j, k, grid, u, v)
    ε̇xx = strain_rate_xx(i, j, k, grid, u)
    ε̇yy = strain_rate_yy(i, j, k, grid, v)
    ε̇xy² = ℑxyᶜᶜᵃ(i, j, k, grid, strain_rate_xy², u, v)
    return ε̇xx^2 + ε̇yy^2 + ε̇xx * ε̇yy + ε̇xy²
end
