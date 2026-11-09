# Ratio of depth-averaged to basal velocity, 1 + βᵇ F₂, at the velocity points.
@inline x_shear_factor(i, j, k, grid, βᵇᵘ, F₂) = @inbounds 1 + βᵇᵘ[i, j, k] * ℑxᶠᵃᵃ(i, j, k, grid, F₂)
@inline y_shear_factor(i, j, k, grid, βᵇᵛ, F₂) = @inbounds 1 + βᵇᵛ[i, j, k] * ℑyᵃᶠᵃ(i, j, k, grid, F₂)

"""
$(TYPEDSIGNATURES)

Compute the basal velocities `uᵇ = ū / (1 + βᵇ F₂)` from the depth-averaged velocities `u`, `v`,
the friction-law drag `βᵇᵘ`, `βᵇᵛ` and the shear integral `F₂`, whose halo must be filled.
"""
function compute_basal_velocities!(uᵇ, vᵇ, u, v, βᵇᵘ, βᵇᵛ, F₂)
    grid = u.grid
    arch = architecture(grid)
    launch!(arch, grid, size(uᵇ), _compute_x_basal_velocity!, uᵇ, u, βᵇᵘ, F₂, grid)
    launch!(arch, grid, size(vᵇ), _compute_y_basal_velocity!, vᵇ, v, βᵇᵛ, F₂, grid)
    return nothing
end

@kernel function _compute_x_basal_velocity!(uᵇ, u, βᵇᵘ, F₂, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds uᵇ[i, j, k] = u[i, j, k] / x_shear_factor(i, j, k, grid, βᵇᵘ, F₂)
end

@kernel function _compute_y_basal_velocity!(vᵇ, v, βᵇᵛ, F₂, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds vᵇ[i, j, k] = v[i, j, k] / y_shear_factor(i, j, k, grid, βᵇᵛ, F₂)
end

"""
$(TYPEDSIGNATURES)

Compute the effective drag `β = βᵇ / (1 + βᵇ F₂)` that the depth-averaged velocity feels,
from the friction-law drag `βᵇᵘ`, `βᵇᵛ` and the shear integral `F₂`, whose halo must be filled.
"""
function compute_effective_drag!(βᵘ, βᵛ, βᵇᵘ, βᵇᵛ, F₂)
    grid = F₂.grid
    arch = architecture(grid)
    launch!(arch, grid, size(βᵘ), _compute_x_effective_drag!, βᵘ, βᵇᵘ, F₂, grid)
    launch!(arch, grid, size(βᵛ), _compute_y_effective_drag!, βᵛ, βᵇᵛ, F₂, grid)
    return nothing
end

@kernel function _compute_x_effective_drag!(βᵘ, βᵇᵘ, F₂, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds βᵘ[i, j, k] = βᵇᵘ[i, j, k] / x_shear_factor(i, j, k, grid, βᵇᵘ, F₂)
end

@kernel function _compute_y_effective_drag!(βᵛ, βᵇᵛ, F₂, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds βᵛ[i, j, k] = βᵇᵛ[i, j, k] / y_shear_factor(i, j, k, grid, βᵇᵛ, F₂)
end

"""
$(TYPEDSIGNATURES)

DIVA counterpart of `update_stress_balance_coefficients!`: from the depth-averaged velocity `x`,
diagnose the basal velocities `uᵇ`, `vᵇ` with the previous `βᵇ` and `F₂`, the friction-law drag
`βᵇᵘ`, `βᵇᵛ` at the bed, the viscosity `ησ` with vertical shear, its column integrals, and the
effective drag `βᵘ`, `βᵛ` passed to `ssa_operator!`. The arguments before `βᵇᵘ` match the SSA update.
"""
function update_diva_coefficients!(x, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, φᵘ, φᵛ, N, μ, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ, walls)
    grid = H.grid
    fill_halo_regions!(x)
    compute_basal_velocities!(uᵇ, vᵇ, x.u, x.v, βᵇᵘ, βᵇᵛ, F₂)
    fill_halo_regions!((uᵇ, vᵇ))
    compute_basal_drag!(βᵇᵘ, βᵇᵛ, friction, uᵇ, vᵇ, φᵘ, φᵛ, N)
    fill_halo_regions!((βᵇᵘ, βᵇᵛ))
    compute_viscosity!(ησ, rheology, x.u, x.v, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ, walls)
    compute_column_integrals!(η̄H, F₂, ησ, H, μ)
    fill_halo_regions!((η̄H, F₂))
    launch!(architecture(grid), grid, size(η̄Hᶠᶠ), _interpolate_to_corners!, η̄Hᶠᶠ, η̄H, grid)
    fill_halo_regions!(η̄Hᶠᶠ)
    compute_effective_drag!(βᵘ, βᵛ, βᵇᵘ, βᵇᵛ, F₂)
    return nothing
end

overwritten_coefficients(::typeof(update_diva_coefficients!), η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, φᵘ, φᵛ, N, μ, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ, args...) =
    (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ)
overwritten_coefficients(::typeof(update_diva_coefficients!), η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, ::Nothing, φᵘ, φᵛ, N, μ, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ, args...) =
    (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, uᵇ, vᵇ)

"""
$(TYPEDSIGNATURES)

Reconstruct the DIVA vertical structure from the depth-averaged velocities: the basal velocities
`uᵇ = ū / (1 + βᵇ F₂)` and the layer velocities `uσ = uᵇ (1 + βᵇ F₁(ζ))`, with the shear integral
`F₁` of `ησ` interpolated to the velocity points. The halos of `ησ`, `H` and `F₂` must be filled.
"""
function compute_velocity_profile!(velocities, βᵇᵘ, βᵇᵛ, F₂, ησ, H)
    (; u, v, uᵇ, vᵇ, uσ, vσ) = velocities
    compute_basal_velocities!(uᵇ, vᵇ, u, v, βᵇᵘ, βᵇᵛ, F₂)
    σgrid = uσ.grid
    arch = architecture(σgrid)
    launch!(arch, σgrid, size(uσ), _compute_x_velocity_profile!, uσ, uᵇ, βᵇᵘ, ησ, H, σgrid)
    launch!(arch, σgrid, size(vσ), _compute_y_velocity_profile!, vσ, vᵇ, βᵇᵛ, ησ, H, σgrid)
    return nothing
end

@kernel function _compute_x_velocity_profile!(uσ, uᵇ, βᵇᵘ, ησ, H, σgrid)
    i, j, k = @index(Global, NTuple)
    F₁ = ℑxᶠᵃᵃ(i, j, k, σgrid, shear_integral, ησ, H)
    @inbounds uσ[i, j, k] = uᵇ[i, j, 1] * (1 + βᵇᵘ[i, j, 1] * F₁)
end

@kernel function _compute_y_velocity_profile!(vσ, vᵇ, βᵇᵛ, ησ, H, σgrid)
    i, j, k = @index(Global, NTuple)
    F₁ = ℑyᵃᶠᵃ(i, j, k, σgrid, shear_integral, ησ, H)
    @inbounds vσ[i, j, k] = vᵇ[i, j, 1] * (1 + βᵇᵛ[i, j, 1] * F₁)
end
