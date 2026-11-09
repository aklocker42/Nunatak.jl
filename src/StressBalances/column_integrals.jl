# Exact integral of (1 - ζ)ᵖ from ζ⁻ to ζ⁺.
@inline power_integral(ζ⁻, ζ⁺, p) = ((1 - ζ⁻)^(p + 1) - (1 - ζ⁺)^(p + 1)) / (p + 1)

# Exact integral of (1 - ζ)ᵖ over layer k.
@inline layer_weight(k, σgrid, p) = power_integral(znode(1, 1, k,   σgrid, Center(), Center(), Face()),
                                                   znode(1, 1, k+1, σgrid, Center(), Center(), Face()), p)

# Exact integral of (1 - ζ)ᵖ over layer k′ ≤ k, up to the center of layer k when k′ = k.
@inline function profile_weight(k′, k, σgrid, p)
    ζ⁻ = znode(1, 1, k′, σgrid, Center(), Center(), Face())
    ζ⁺ = ifelse(k′ == k, znode(1, 1, k, σgrid, Center(), Center(), Center()),
                          znode(1, 1, k′+1, σgrid, Center(), Center(), Face()))
    return power_integral(ζ⁻, ζ⁺, p)
end

# ∫₀^ζₖ f (1 - ζ′)ᵖ dζ′ from the bed to the center of layer k, exact for layer-wise constant f.
@inline function profile_integral(f, i, j, k, σgrid, p, args...)
    I = zero(σgrid)
    for k′ in 1:k
        I += f(i, j, k′, args...) * profile_weight(k′, k, σgrid, p)
    end
    return I
end

# ∫₀¹ f (1 - ζ)ᵖ dζ over the column, exact for layer-wise constant f.
@inline function column_integral(f, i, j, σgrid, p, args...)
    I = zero(σgrid)
    for k in 1:size(σgrid, 3)
        I += f(i, j, k, args...) * layer_weight(k, σgrid, p)
    end
    return I
end

@inline inverse_viscosity(i, j, k, ησ) = @inbounds 1 / ησ[i, j, k]

"""
$(TYPEDSIGNATURES)

Shear integral `F₁(ζₖ) = H ∫₀^ζₖ (1 - ζ′) / η dζ′` from the bed to the center `ζₖ` of layer `k`,
so the DIVA velocity there is `uᵇ (1 + βᵇ F₁)`. Depth-averaging `F₁` gives `F₂`.
"""
@inline shear_integral(i, j, k, σgrid, ησ, H) = @inbounds profile_integral(inverse_viscosity, i, j, k, σgrid, 1, ησ) * H[i, j, 1]

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
        F += inverse_viscosity(i, j, k, ησ) * layer_weight(k, σgrid, 2)
    end
    @inbounds η̄H[i, j, 1] = η̄ * H[i, j, 1] * ice_mask_at(i, j, 1, μ)
    @inbounds F₂[i, j, 1] = F * H[i, j, 1]
end
