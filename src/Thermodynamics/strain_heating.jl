"""
$(TYPEDSIGNATURES)

Compute the strain heating `Φ = τ:ε̇ = 4 η ε̇ₑ²` from the layer viscosity and the physical strain rate, without the
regularisation `ε̇₀` of the flow law. Under SSA the strain rate is the membrane part. Velocity halos must be filled.
"""
function compute_strain_heating!(Φ, model, ::SSA)
    (; grid, σgrid) = model
    launch!(architecture(grid), σgrid, :xyz, _compute_membrane_heating!, Φ, grid, model.velocities.u, model.velocities.v, model.auxiliary.ησ)
    return nothing
end

"""
$(TYPEDSIGNATURES)

DIVA strain heating adds the vertical shear to the membrane part, `4 η (ε̇xz² + ε̇yz²) = τᵇ² (1 - ζ)² / η`. The factor
`(1 - ζ)²` is its exact mean over the layer, so the column integral of the shear part is `τᵇ² F₂`, the work
`τᵇ⋅(ū - uᵇ)` that the depth-averaged balance attributes to shear.
"""
function compute_strain_heating!(Φ, model, ::DIVA)
    (; grid, σgrid) = model
    (; u, v, uᵇ, vᵇ) = model.velocities
    (; ησ, βᵇᵘ, βᵇᵛ) = model.auxiliary
    launch!(architecture(grid), σgrid, :xyz, _compute_shear_heating!, Φ, grid, σgrid, u, v, uᵇ, vᵇ, βᵇᵘ, βᵇᵛ, ησ)
    return nothing
end

@kernel function _compute_membrane_heating!(Φ, grid, u, v, ησ)
    i, j, k = @index(Global, NTuple)
    @inbounds Φ[i, j, k] = 4 * ησ[i, j, k] * effective_strain_rate²(i, j, 1, grid, u, v)
end

@inline basal_stress²(i, j, k, grid, β, u) = @inbounds (β[i, j, k] * u[i, j, k])^2

@kernel function _compute_shear_heating!(Φ, grid, σgrid, u, v, uᵇ, vᵇ, βᵇᵘ, βᵇᵛ, ησ)
    i, j, k = @index(Global, NTuple)
    η = @inbounds ησ[i, j, k]
    τᵇ² = ℑxᶜᵃᵃ(i, j, 1, grid, basal_stress², βᵇᵘ, uᵇ) + ℑyᵃᶜᵃ(i, j, 1, grid, basal_stress², βᵇᵛ, vᵇ)
    shear = τᵇ² * shear_weight(k, σgrid) / layer_thickness(k, σgrid) / ifelse(iszero(η), typemax(η), η)
    @inbounds Φ[i, j, k] = 4 * η * effective_strain_rate²(i, j, 1, grid, u, v) + shear
end
