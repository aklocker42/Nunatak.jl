# Exact integral of (1 - ζ)² over layer k, so F₂ is exact for layer-wise constant η.
@inline function shear_weight(k, σgrid)
    ζ⁻ = znode(1, 1, k,   σgrid, Center(), Center(), Face())
    ζ⁺ = znode(1, 1, k+1, σgrid, Center(), Center(), Face())
    return ((1 - ζ⁻)^3 - (1 - ζ⁺)^3) / 3
end

# Exact integral of (1 - ζ) from the bed to the centre of layer k, split into full layers below and the lower half of k.
@inline function profile_weight(k′, k, σgrid)
    ζ⁻ = znode(1, 1, k′, σgrid, Center(), Center(), Face())
    ζ⁺ = ifelse(k′ == k, znode(1, 1, k, σgrid, Center(), Center(), Center()),
                          znode(1, 1, k′+1, σgrid, Center(), Center(), Face()))
    return ((1 - ζ⁻)^2 - (1 - ζ⁺)^2) / 2
end

"""
$(TYPEDSIGNATURES)

Shear integral `F₁(ζₖ) = H ∫₀^ζₖ (1 - ζ′) / η dζ′` from the bed to the centre `ζₖ` of layer `k`,
so the DIVA velocity there is `uᵇ (1 + βᵇ F₁)`. Depth-averaging `F₁` gives `F₂`.
"""
@inline function shear_integral(i, j, k, σgrid, ησ, H)
    F = zero(eltype(ησ))
    for k′ in 1:k
        @inbounds F += profile_weight(k′, k, σgrid) / ησ[i, j, k′]
    end
    return @inbounds F * H[i, j, 1]
end

@inline layer_thickness(k, σgrid) = znode(1, 1, k+1, σgrid, Center(), Center(), Face()) -
                                    znode(1, 1, k,   σgrid, Center(), Center(), Face())

"""
$(TYPEDSIGNATURES)

Compute the depth-integrated viscosity `η̄H = H ∫₀¹ η dζ` and the DIVA shear integral
`F₂ = H ∫₀¹ (1 - ζ)² / η dζ` from the column viscosity `ησ` on the σ grid.
For uniform `η`, `η̄H = η H` and `F₂ = H / (3η)`. Given an ice mask `μ`, `η̄H` is scaled by it.
"""
function compute_column_integrals!(η̄H, F₂, ησ, H, μ = nothing)
    grid = H.grid
    σgrid = ησ.grid
    launch!(architecture(grid), grid, :xy, _compute_column_integrals!, η̄H, F₂, ησ, H, μ, σgrid)
    return nothing
end

@kernel function _compute_column_integrals!(η̄H, F₂, ησ, H, μ, σgrid)
    i, j = @index(Global, NTuple)
    Nσ = size(σgrid, 3)
    η̄ = zero(eltype(η̄H))
    F = zero(eltype(F₂))
    for k in 1:Nσ
        @inbounds η̄ += ησ[i, j, k] * layer_thickness(k, σgrid)
        @inbounds F += shear_weight(k, σgrid) / ησ[i, j, k]
    end
    @inbounds η̄H[i, j, 1] = η̄ * H[i, j, 1] * ice_mask_at(i, j, 1, μ)
    @inbounds F₂[i, j, 1] = F * H[i, j, 1]
end
