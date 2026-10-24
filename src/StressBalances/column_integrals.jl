using Oceananigans.Grids: znode

# Exact integral of (1 - ζ)² over layer k, so F₂ is exact for layer-wise constant η.
@inline function shear_weight(k, σgrid)
    ζ⁻ = znode(1, 1, k,   σgrid, Center(), Center(), Face())
    ζ⁺ = znode(1, 1, k+1, σgrid, Center(), Center(), Face())
    return ((1 - ζ⁻)^3 - (1 - ζ⁺)^3) / 3
end

@inline layer_thickness(k, σgrid) = znode(1, 1, k+1, σgrid, Center(), Center(), Face()) -
                                    znode(1, 1, k,   σgrid, Center(), Center(), Face())

"""
$(TYPEDSIGNATURES)

Compute the depth-integrated viscosity `η̄H = H ∫₀¹ η dζ` and the DIVA shear integral
`F₂ = H ∫₀¹ (1 - ζ)² / η dζ` from the column viscosity `ησ` on the σ grid.
For uniform `η`, `η̄H = η H` and `F₂ = H / (3η)`.
"""
function compute_column_integrals!(η̄H, F₂, ησ, H)
    grid = H.grid
    σgrid = ησ.grid
    launch!(architecture(grid), grid, :xy, _compute_column_integrals!, η̄H, F₂, ησ, H, σgrid)
    return nothing
end

@kernel function _compute_column_integrals!(η̄H, F₂, ησ, H, σgrid)
    i, j = @index(Global, NTuple)
    Nσ = size(σgrid, 3)
    η̄ = zero(eltype(η̄H))
    F = zero(eltype(F₂))
    for k in 1:Nσ
        @inbounds η̄ += ησ[i, j, k] * layer_thickness(k, σgrid)
        @inbounds F += shear_weight(k, σgrid) / ησ[i, j, k]
    end
    @inbounds η̄H[i, j, 1] = η̄ * H[i, j, 1]
    @inbounds F₂[i, j, 1] = F * H[i, j, 1]
end
