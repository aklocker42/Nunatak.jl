"""
$(TYPEDSIGNATURES)

Freeze the stress-balance coefficients at the velocity `x`: Glen viscosity `ησ`, its column
integrals `η̄H` and `F₂`, `η̄H` interpolated to the cell corners `η̄Hᶠᶠ`, and the basal drag
`βᵘ`, `βᵛ` scaled by the grounded fractions `φᵘ`, `φᵛ` (`N` is the effective pressure). `η̄H` is scaled
by the ice mask `μ` when it is given, and `walls` sets the shear at wall corners. The argument list
after `x` is the one `ssa_operator!` takes, so this is the `update_coefficients!` of `picard_solve!`.
"""
function update_stress_balance_coefficients!(x, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, φᵘ, φᵛ, N, μ, walls)
    grid = H.grid
    fill_halo_regions!(x)
    compute_viscosity!(ησ, rheology, x.u, x.v, walls)
    compute_column_integrals!(η̄H, F₂, ησ, H, μ)
    fill_halo_regions!(η̄H)
    launch!(architecture(grid), grid, size(η̄Hᶠᶠ), _interpolate_to_corners!, η̄Hᶠᶠ, η̄H, grid)
    fill_halo_regions!(η̄Hᶠᶠ)
    compute_basal_drag!(βᵘ, βᵛ, friction, x.u, x.v, φᵘ, φᵛ, N)
    return nothing
end

# The coefficients the update overwrites, which the fixed-point Picard adjoint carries; without friction the drag is left untouched.
overwritten_coefficients(::typeof(update_stress_balance_coefficients!), η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, args...) = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂)
overwritten_coefficients(::typeof(update_stress_balance_coefficients!), η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, ::Nothing, args...) = (η̄H, η̄Hᶠᶠ, ησ, F₂)

@kernel function _interpolate_to_corners!(η̄Hᶠᶠ, η̄H, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds η̄Hᶠᶠ[i, j, k] = ℑxyᶠᶠᵃ(i, j, k, grid, η̄H)
end
