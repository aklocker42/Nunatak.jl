using Oceananigans.TimeSteppers: SplitRungeKuttaTimeStepper

function Oceananigans.TimeSteppers.cache_current_fields!(model::IceSheetModel)
    H⁻ = model.timestepper.Ψ⁻.H
    parent(H⁻) .= parent(model.geometry.H)
    return nothing
end

"""
$(TYPEDSIGNATURES)

One split Runge-Kutta stage: `H = H⁻ + Δτ (a − m)`, clipped at zero. Thickness advection by
the velocity field enters in phase 2.
"""
function Oceananigans.TimeSteppers.rk_substep!(model::IceSheetModel, Δτ, callbacks)
    grid = model.grid
    launch!(architecture(grid), grid, :xy, _thickness_substep!,
            model.geometry.H, model.timestepper.Ψ⁻.H, model.surface_mass_balance, model.basal_melt_rate, Δτ)
    return nothing
end

@kernel function _thickness_substep!(H, H⁻, a, m, Δτ)
    i, j = @index(Global, NTuple)
    @inbounds H[i, j, 1] = max(zero(eltype(H)), H⁻[i, j, 1] + Δτ * (a[i, j, 1] - m[i, j, 1]))
end
