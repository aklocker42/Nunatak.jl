"""
$(TYPEDEF)

Jacobi preconditioner for `ssa_operator!` that evaluates the operator diagonal from the
current coefficients at each application, so it stays consistent through Picard iterations
without storage. Wall and ice-free rows, and rows with neither viscosity nor drag, have unit diagonal. The wall
condition among the trailing `args` sets the shear term at wall corners, as in `ssa_operator!`.
"""
struct StressBalanceDiagonalPreconditioner end

function precondition!(z::VelocityVector, ::StressBalanceDiagonalPreconditioner, r::VelocityVector, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, args...)
    grid = η̄H.grid
    arch = architecture(grid)
    walls = side_walls(args...)
    launch!(arch, grid, size(z.u), _x_diagonal_precondition!, z.u, r.u, η̄H, η̄Hᶠᶠ, βᵘ, walls, grid)
    launch!(arch, grid, size(z.v), _y_diagonal_precondition!, z.v, r.v, η̄H, η̄Hᶠᶠ, βᵛ, walls, grid)
    return nothing
end

@inline x_normal_diagonal(i, j, k, grid, η̄H) = @inbounds 4 * Δyᶜᶜᶜ(i, j, k, grid) * η̄H[i, j, k] / Δxᶜᶜᶜ(i, j, k, grid)
@inline y_normal_diagonal(i, j, k, grid, η̄H) = @inbounds 4 * Δxᶜᶜᶜ(i, j, k, grid) * η̄H[i, j, k] / Δyᶜᶜᶜ(i, j, k, grid)

@inline wall_corner(i, j, k, grid) = peripheral_node(i, j, k, grid, Face(), Face(), Center())

@inline corner_shear_factor(i, j, k, grid, walls) = ifelse(wall_corner(i, j, k, grid), wall_shear_factor(walls, grid), one(grid))

@inline x_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ, walls) =
    @inbounds corner_shear_factor(i, j, k, grid, walls) * Δxᶠᶠᶜ(i, j, k, grid) * η̄Hᶠᶠ[i, j, k] / Δyᶠᶠᶜ(i, j, k, grid)
@inline y_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ, walls) =
    @inbounds corner_shear_factor(i, j, k, grid, walls) * Δyᶠᶠᶜ(i, j, k, grid) * η̄Hᶠᶠ[i, j, k] / Δxᶠᶠᶜ(i, j, k, grid)

@inline function x_diagonal(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵘ, walls)
    d = @inbounds Azᶠᶜᶜ(i, j, k, grid) * βᵘ[i, j, k] +
        x_normal_diagonal(i, j, k, grid, η̄H) + x_normal_diagonal(i-1, j, k, grid, η̄H) +
        x_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ, walls) + x_shear_diagonal(i, j+1, k, grid, η̄Hᶠᶠ, walls)
    return ifelse(inactiveᶠᶜᶜ(i, j, k, grid, η̄H) | iszero(d), one(grid), d)
end

@inline function y_diagonal(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵛ, walls)
    d = @inbounds Azᶜᶠᶜ(i, j, k, grid) * βᵛ[i, j, k] +
        y_normal_diagonal(i, j, k, grid, η̄H) + y_normal_diagonal(i, j-1, k, grid, η̄H) +
        y_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ, walls) + y_shear_diagonal(i+1, j, k, grid, η̄Hᶠᶠ, walls)
    return ifelse(inactiveᶜᶠᶜ(i, j, k, grid, η̄H) | iszero(d), one(grid), d)
end

@kernel function _x_diagonal_precondition!(zu, ru, η̄H, η̄Hᶠᶠ, βᵘ, walls, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds zu[i, j, k] = ru[i, j, k] / x_diagonal(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵘ, walls)
end

@kernel function _y_diagonal_precondition!(zv, rv, η̄H, η̄Hᶠᶠ, βᵛ, walls, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds zv[i, j, k] = rv[i, j, k] / y_diagonal(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵛ, walls)
end
