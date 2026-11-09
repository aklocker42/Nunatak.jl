function cache_current_fields!(model::IceSheetModel)
    H⁻ = model.timestepper.Ψ⁻.H
    parent(H⁻) .= parent(model.geometry.H)
    return nothing
end

"""
$(TYPEDSIGNATURES)

One split Runge-Kutta stage of the mass balance ∂H/∂t + ∇⋅(ūH) = a − m:
the tendency at the current stage is stored in the time stepper, then `H = H⁻ + Δτ G`, clipped at zero.
The thickness added by the clip in the final stage is accumulated in `clipped_thickness`. After the final stage
the calving front advances when the masking is a `PartialCellFront`; the final stage spans the whole step.
"""
function rk_substep!(model::IceSheetModel, Δτ, callbacks)
    grid = model.grid
    G = model.timestepper.Gⁿ.H
    final_stage = model.clock.stage == model.timestepper.Nstages
    compute_thickness_tendency!(G, model)
    launch!(architecture(grid), grid, :xy, _thickness_substep!, model.geometry.H, model.geometry.clipped_thickness,
            model.timestepper.Ψ⁻.H, G, Δτ, final_stage)
    final_stage && advance_front!(model.masking, model, Δτ)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Compute `G = a − m − mᵍ − ∇⋅(ūH)` with the model's advection scheme and the depth-averaged velocities.
With a grounding-line treatment the melt acts on the floating part only, `(1 - φ) m`. The grounded melt `mᵍ`,
the change of the basal water layer per unit time, comes from the thermodynamics when they are coupled.
With a `PartialCellFront` the sources here act on full cells only, front cells pass no thickness on, and
`advance_front!` applies the sources to the partially filled front cells.
"""
function compute_thickness_tendency!(G, model::IceSheetModel)
    grid = model.grid
    update_bounds_preserving_limiter!(model.advection, grid, model.geometry.H)
    launch!(architecture(grid), grid, :xy, _compute_thickness_tendency!, G, grid, model.advection,
            advecting_velocities(model), model.geometry.H, model.surface_mass_balance, model.basal_melt_rate,
            floating_melt_fraction(model.grounding, model.geometry), coupled_grounded_melt(model.thermodynamics),
            front_mask(model.masking, model.geometry))
    return nothing
end

advecting_velocities(model::IceSheetModel) = (u = model.velocities.u, v = model.velocities.v, w = ZeroField())

floating_melt_fraction(::Nothing, geometry) = nothing
floating_melt_fraction(grounding, geometry) = geometry.φ

@inline basal_melt(i, j, m, ::Nothing) = @inbounds m[i, j, 1]
@inline basal_melt(i, j, m, φ) = @inbounds (1 - φ[i, j, 1]) * m[i, j, 1]

coupled_grounded_melt(::Nothing) = nothing
coupled_grounded_melt(thermodynamics) = thermodynamics.grounded_melt_coupling ? thermodynamics.state.mᵍ : nothing

@inline grounded_melt(i, j, ::Nothing) = zero(i)
@inline grounded_melt(i, j, mᵍ) = @inbounds mᵍ[i, j, 1]

@kernel function _compute_thickness_tendency!(G, grid, advection, U, H, a, m, φ, mᵍ, μ)
    i, j = @index(Global, NTuple)
    S = @inbounds a[i, j, 1] - basal_melt(i, j, m, φ) - grounded_melt(i, j, mᵍ)
    @inbounds G[i, j, 1] = ifelse(ice_source_mask(i, j, μ), S, zero(S)) - thickness_flux_divergence(i, j, 1, grid, advection, U, H, μ)
end

@kernel function _thickness_substep!(H, Hᶜ, H⁻, G, Δτ, final_stage)
    i, j = @index(Global, NTuple)
    @inbounds begin
        H★ = H⁻[i, j, 1] + Δτ * G[i, j, 1]
        H[i, j, 1] = max(zero(H★), H★)
        Hᶜ[i, j, 1] += ifelse(final_stage, max(zero(H★), -H★), zero(H★))
    end
end

"""
$(TYPEDSIGNATURES)

Shortest time for the depth-averaged flow to cross a cell, used by `TimeStepWizard`.
"""
cell_advection_timescale(model::IceSheetModel) = cell_advection_timescale(model.grid, advecting_velocities(model))

"""
$(TYPEDSIGNATURES)

Shortest explicit-diffusion time `Δ² / D` of the shallow-ice thickness flux, used by `TimeStepWizard` with
`diffusive_cfl`; infinite for the depth-integrated stress balances, whose flux is advective. As in Oceananigans
this is a per-direction bound. The linearised shallow-ice diffusivity exceeds `D` and both directions add up,
so two-dimensional runs need a `diffusive_cfl` well below 1/2; the Halfar test uses 1/4.
"""
cell_diffusion_timescale(model::IceSheetModel) = cell_diffusion_timescale(model, model.stress_balance)
cell_diffusion_timescale(model::IceSheetModel, stress_balance) = convert(eltype(model.grid), Inf)

function cell_diffusion_timescale(model::IceSheetModel, ::ShallowIce)
    grid = model.grid
    (; H, s) = model.geometry
    rateˣ = KernelFunctionOperation{Face, Center, Center}(x_diffusion_rate, grid, model.velocities.u, H, s)
    rateʸ = KernelFunctionOperation{Center, Face, Center}(y_diffusion_rate, grid, model.velocities.v, H, s)
    return 1 / max(maximum(rateˣ), maximum(rateʸ))
end
