# Walls carry no flux, whatever the velocity field holds there.
@inline layer_velocityᶠᶜᶜ(i, j, k, grid, uσ) = @inbounds ifelse(peripheral_node(i, j, 1, grid, Face(), Center(), Center()), zero(grid), uσ[i, j, k])
@inline layer_velocityᶜᶠᶜ(i, j, k, grid, vσ) = @inbounds ifelse(peripheral_node(i, j, 1, grid, Center(), Face(), Center()), zero(grid), vσ[i, j, k])

@inline upwind_value(u, c⁻, c⁺) = ifelse(u > 0, c⁻, c⁺)

# Upwind volume flux per unit Δζ through the west and south faces of a cell.
@inline function x_volume_flux(i, j, k, grid, uσ, H)
    u = layer_velocityᶠᶜᶜ(i, j, k, grid, uσ)
    return @inbounds Δyᶠᶜᶜ(i, j, 1, grid) * u * upwind_value(u, H[i-1, j, 1], H[i, j, 1])
end

@inline function y_volume_flux(i, j, k, grid, vσ, H)
    v = layer_velocityᶜᶠᶜ(i, j, k, grid, vσ)
    return @inbounds Δxᶜᶠᶜ(i, j, 1, grid) * v * upwind_value(v, H[i, j-1, 1], H[i, j, 1])
end

@inline x_enthalpy_flux(i, j, k, grid, uσ, H, E) = @inbounds x_volume_flux(i, j, k, grid, uσ, H) * upwind_value(uσ[i, j, k], E[i-1, j, k], E[i, j, k])
@inline y_enthalpy_flux(i, j, k, grid, vσ, H, E) = @inbounds y_volume_flux(i, j, k, grid, vσ, H) * upwind_value(vσ[i, j, k], E[i, j-1, k], E[i, j, k])

@inline function volume_flux_divergence(i, j, k, grid, uσ, vσ, H)
    return (x_volume_flux(i+1, j, k, grid, uσ, H) - x_volume_flux(i, j, k, grid, uσ, H) +
            y_volume_flux(i, j+1, k, grid, vσ, H) - y_volume_flux(i, j, k, grid, vσ, H)) / Azᶜᶜᶜ(i, j, 1, grid)
end

@inline function enthalpy_flux_divergence(i, j, k, grid, uσ, vσ, H, E)
    return (x_enthalpy_flux(i+1, j, k, grid, uσ, H, E) - x_enthalpy_flux(i, j, k, grid, uσ, H, E) +
            y_enthalpy_flux(i, j+1, k, grid, vσ, H, E) - y_enthalpy_flux(i, j, k, grid, vσ, H, E)) / Azᶜᶜᶜ(i, j, 1, grid)
end

# Mass balance of layer k: Δζ ∂ₜH + Δζ ∇⋅(H u) + W[k+1] - W[k] = 0, with W[1] = -m - mᵍ. W is the upward volume flux
# across the layer face relative to the moving σ surface, and is exact for the thickness change the column actually had.
# The coupled grounded melt mᵍ is the one the thickness used since the last energy step; drainage leaves through the base.
@kernel function _compute_layer_volume_fluxes!(W, grid, σgrid, H, Hⁿ, uσ, vσ, m, φ, mᵍ, parameters, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Nσ = size(σgrid, 3)
        ∂ₜH = (column_thickness(i, j, H, parameters) - column_thickness(i, j, Hⁿ, parameters)) / Δt
        W[i, j, 1] = -basal_melt(i, j, m, φ) - grounded_melt(i, j, mᵍ)
        for k in 1:Nσ
            W[i, j, k+1] = W[i, j, k] - layer_thickness(k, σgrid) * (∂ₜH + volume_flux_divergence(i, j, k, grid, uσ, vσ, Hⁿ))
        end
    end
end
