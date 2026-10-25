function cache_current_fields!(model::IceSheetModel)
    H⁻ = model.timestepper.Ψ⁻.H
    parent(H⁻) .= parent(model.geometry.H)
    return nothing
end

"""
$(TYPEDSIGNATURES)

One split Runge-Kutta stage of the mass balance ∂H/∂t + ∇⋅(ūH) = a − m:
the tendency at the current stage is stored in the time stepper, then `H = H⁻ + Δτ G`, clipped at zero.
"""
function rk_substep!(model::IceSheetModel, Δτ, callbacks)
    grid = model.grid
    G = model.timestepper.Gⁿ.H
    compute_thickness_tendency!(G, model)
    launch!(architecture(grid), grid, :xy, _thickness_substep!, model.geometry.H, model.timestepper.Ψ⁻.H, G, Δτ)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute `G = a − m − ∇⋅(ūH)` with the model's advection scheme and the depth-averaged velocities.
"""
function compute_thickness_tendency!(G, model::IceSheetModel)
    grid = model.grid
    update_bounds_preserving_limiter!(model.advection, grid, model.geometry.H)
    launch!(architecture(grid), grid, :xy, _compute_thickness_tendency!, G, grid, model.advection,
            advecting_velocities(model), model.geometry.H, model.surface_mass_balance, model.basal_melt_rate)
    return nothing
end

advecting_velocities(model::IceSheetModel) = (u = model.velocities.u, v = model.velocities.v, w = ZeroField())

@kernel function _compute_thickness_tendency!(G, grid, advection, U, H, a, m)
    i, j = @index(Global, NTuple)
    @inbounds G[i, j, 1] = a[i, j, 1] - m[i, j, 1] - div_Uc(i, j, 1, grid, advection, U, H)
end

@kernel function _thickness_substep!(H, H⁻, G, Δτ)
    i, j = @index(Global, NTuple)
    @inbounds H[i, j, 1] = max(zero(eltype(H)), H⁻[i, j, 1] + Δτ * G[i, j, 1])
end

"""
$(TYPEDSIGNATURES)

Shortest time for the depth-averaged flow to cross a cell, used by `TimeStepWizard`.
"""
cell_advection_timescale(model::IceSheetModel) = cell_advection_timescale(model.grid, advecting_velocities(model))
