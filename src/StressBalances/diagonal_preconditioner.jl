"""
$(TYPEDEF)

Jacobi preconditioner for `ssa_operator!` that evaluates the operator diagonal from the
current coefficients at each application, so it stays consistent through Picard iterations
without storage. Wall rows have unit diagonal.
"""
struct StressBalanceDiagonalPreconditioner end

function precondition!(z::VelocityVector, ::StressBalanceDiagonalPreconditioner, r::VelocityVector, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, args...)
    grid = η̄H.grid
    arch = architecture(grid)
    launch!(arch, grid, size(z.u), _x_diagonal_precondition!, z.u, r.u, η̄H, η̄Hᶠᶠ, βᵘ, grid)
    launch!(arch, grid, size(z.v), _y_diagonal_precondition!, z.v, r.v, η̄H, η̄Hᶠᶠ, βᵛ, grid)
    return nothing
end

@inline x_normal_diagonal(i, j, k, grid, η̄H) = @inbounds 4 * Δyᶜᶜᶜ(i, j, k, grid) * η̄H[i, j, k] / Δxᶜᶜᶜ(i, j, k, grid)
@inline y_normal_diagonal(i, j, k, grid, η̄H) = @inbounds 4 * Δxᶜᶜᶜ(i, j, k, grid) * η̄H[i, j, k] / Δyᶜᶜᶜ(i, j, k, grid)

@inline x_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ) = @inbounds Δxᶠᶠᶜ(i, j, k, grid) * η̄Hᶠᶠ[i, j, k] / Δyᶠᶠᶜ(i, j, k, grid)
@inline y_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ) = @inbounds Δyᶠᶠᶜ(i, j, k, grid) * η̄Hᶠᶠ[i, j, k] / Δxᶠᶠᶜ(i, j, k, grid)

@inline function x_diagonal(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵘ)
    d = @inbounds Azᶠᶜᶜ(i, j, k, grid) * βᵘ[i, j, k] +
        x_normal_diagonal(i, j, k, grid, η̄H) + x_normal_diagonal(i-1, j, k, grid, η̄H) +
        x_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ) + x_shear_diagonal(i, j+1, k, grid, η̄Hᶠᶠ)
    return ifelse(peripheral_node(i, j, k, grid, Face(), Center(), Center()), one(grid), d)
end

@inline function y_diagonal(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵛ)
    d = @inbounds Azᶜᶠᶜ(i, j, k, grid) * βᵛ[i, j, k] +
        y_normal_diagonal(i, j, k, grid, η̄H) + y_normal_diagonal(i, j-1, k, grid, η̄H) +
        y_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ) + y_shear_diagonal(i+1, j, k, grid, η̄Hᶠᶠ)
    return ifelse(peripheral_node(i, j, k, grid, Center(), Face(), Center()), one(grid), d)
end

@kernel function _x_diagonal_precondition!(zu, ru, η̄H, η̄Hᶠᶠ, βᵘ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds zu[i, j, k] = ru[i, j, k] / x_diagonal(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵘ)
end

@kernel function _y_diagonal_precondition!(zv, rv, η̄H, η̄Hᶠᶠ, βᵛ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds zv[i, j, k] = rv[i, j, k] / y_diagonal(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵛ)
end
