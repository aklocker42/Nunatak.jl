"""
$(TYPEDSIGNATURES)

Freeze the stress-balance coefficients at the velocity `x`: Glen viscosity `ησ`, its column
integrals `η̄H` and `F₂`, and `η̄H` interpolated to the cell corners `η̄Hᶠᶠ`. The argument list
after `x` is the one `ssa_operator!` takes, so this is the `update_coefficients!` of `picard_solve!`.
"""
function update_stress_balance_coefficients!(x, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology)
    grid = H.grid
    fill_halo_regions!(x)
    compute_viscosity!(ησ, rheology, x.u, x.v)
    compute_column_integrals!(η̄H, F₂, ησ, H)
    fill_halo_regions!(η̄H)
    launch!(architecture(grid), grid, size(η̄Hᶠᶠ), _interpolate_to_corners!, η̄Hᶠᶠ, η̄H, grid)
    fill_halo_regions!(η̄Hᶠᶠ)
    return nothing
end

@kernel function _interpolate_to_corners!(η̄Hᶠᶠ, η̄H, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds η̄Hᶠᶠ[i, j, k] = ℑxyᶠᶠᵃ(i, j, k, grid, η̄H)
end
