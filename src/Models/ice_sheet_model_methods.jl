const ISM = IceSheetModel

"""
$(TYPEDSIGNATURES)

Set the ice thickness `H` and the depth-averaged velocities `u`, `v`, then update the state.
Given velocities are the initial guess of the stress-balance solve.
"""
function set!(model::ISM; H = nothing, u = nothing, v = nothing)
    set_if_given!(model.geometry.H, H)
    set_if_given!(model.velocities.u, u)
    set_if_given!(model.velocities.v, v)
    update_state!(model)
    return nothing
end

set_if_given!(field, ::Nothing) = nothing
set_if_given!(field, value) = set!(field, value)

set!(model::ISM, new_clock::Clock) = set!(model.clock, new_clock)

architecture(model::ISM) = model.architecture
iteration(model::ISM) = model.clock.iteration
prettytime(model::ISM) = prettytime(model.clock.time)
initialize!(::ISM) = nothing
reset!(::ISM) = nothing
default_included_properties(::ISM) = [:grid]
checkpointer_address(::ISM) = "IceSheetModel"

fields(model::ISM) = (; u = model.velocities.u, v = model.velocities.v, model.geometry...,
                                     b = model.bed, a = model.surface_mass_balance, m = model.basal_melt_rate)

prognostic_fields(model::ISM) = (; H = model.geometry.H)

"""
$(TYPEDSIGNATURES)

Diagnose the geometry (`s`, `φ`, `φᵘ`, `φᵛ`, `N`, `ice_mask`) from the thickness and bed, then solve the
stress balance for the velocities.
"""
function update_state!(model::ISM, callbacks = [])
    fill_halo_regions!(model.geometry.H, model.clock, fields(model))
    update_geometry!(model.geometry, model.grounding, model.masking, model.bed, model.constants)
    update_velocities!(model, model.velocity_solver)
    fill_halo_regions!((model.velocities.u, model.velocities.v), model.clock, fields(model))
    return nothing
end

function update_geometry!(geometry, grounding, masking, b, constants)
    (; H, s, φ, φᵘ, φᵛ, N, ice_mask) = geometry
    compute_surface_elevation!(s, grounding, H, b, constants)
    compute_grounded_fractions!(φ, φᵘ, φᵛ, grounding, H, b, constants)
    compute_effective_pressure!(N, H, b, constants)
    compute_ice_mask!(ice_mask, masking, H)
    fill_halo_regions!((s, N, ice_mask))
    return nothing
end

update_velocities!(model, ::Nothing) = nothing
update_velocities!(model, solver::StressBalanceSolver) = update_velocities!(model, solver, model.stress_balance)

function update_velocities!(model, solver::StressBalanceSolver, ::SSA)
    (; H, s, φᵘ, φᵛ, N, ice_mask) = model.geometry
    (; ησ, η̄H, η̄Hᶠᶠ, F₂, βᵘ, βᵛ, τᵘ, τᵛ) = model.auxiliary
    constants = model.constants
    τ = VelocityVector(τᵘ, τᵛ)
    x = VelocityVector(model.velocities.u, model.velocities.v)
    compute_driving_stress!(τ, H, s, constants.ice_density, constants.gravitational_acceleration, ice_mask)
    coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, model.rheology, model.friction, φᵘ, φᵛ, N, ice_mask)
    solver.iteration = picard_solve!(x, solver.picard, solver.linear_solver, update_stress_balance_coefficients!, τ, coefficients...)
    extend_depth_average!(model.velocities)
    return nothing
end

function update_velocities!(model, solver::StressBalanceSolver, ::DIVA)
    (; H, s, φᵘ, φᵛ, N, ice_mask) = model.geometry
    (; ησ, η̄H, η̄Hᶠᶠ, F₂, βᵘ, βᵛ, βᵇᵘ, βᵇᵛ, τᵘ, τᵛ) = model.auxiliary
    (; uᵇ, vᵇ) = model.velocities
    constants = model.constants
    τ = VelocityVector(τᵘ, τᵛ)
    x = VelocityVector(model.velocities.u, model.velocities.v)
    compute_driving_stress!(τ, H, s, constants.ice_density, constants.gravitational_acceleration, ice_mask)
    coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, model.rheology, model.friction, φᵘ, φᵛ, N, ice_mask, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ)
    solver.iteration = picard_solve!(x, solver.picard, solver.linear_solver, update_diva_coefficients!, τ, coefficients...)
    fill_halo_regions!(ησ)
    compute_velocity_profile!(model.velocities, βᵇᵘ, βᵇᵛ, F₂, ησ, H)
    return nothing
end

# Under SSA the velocity is uniform in the column.
function extend_depth_average!(velocities)
    (; u, v, uᵇ, vᵇ, uσ, vσ) = velocities
    parent(uᵇ) .= parent(u)
    parent(vᵇ) .= parent(v)
    σgrid = uσ.grid
    launch!(architecture(σgrid), σgrid, size(uσ), _extend_depth_average!, uσ, u)
    launch!(architecture(σgrid), σgrid, size(vσ), _extend_depth_average!, vσ, v)
    return nothing
end

@kernel function _extend_depth_average!(uσ, u)
    i, j, k = @index(Global, NTuple)
    @inbounds uσ[i, j, k] = u[i, j, 1]
end

function prognostic_state(model::ISM)
    return (clock = prognostic_state(model.clock),
            geometry = prognostic_state(model.geometry),
            velocities = prognostic_state(model.velocities),
            timestepper = prognostic_state(model.timestepper))
end

function restore_prognostic_state!(restored::ISM, from)
    restore_prognostic_state!(restored.clock, from.clock)
    restore_prognostic_state!(restored.geometry, from.geometry)
    restore_prognostic_state!(restored.velocities, from.velocities)
    restore_prognostic_state!(restored.timestepper, from.timestepper)
    return restored
end

restore_prognostic_state!(::ISM, ::Nothing) = nothing

function Base.summary(model::ISM)
    A = Base.summary(architecture(model))
    G = nameof(typeof(model.grid))
    return string("IceSheetModel{$A, $G}",
                  "(time = ", prettytime(model.clock.time),
                  ", iteration = ", prettysummary(model.clock.iteration), ")")
end

function Base.show(io::IO, model::ISM)
    print(io, summary(model), '\n')
    print(io, "├── grid: ", summary(model.grid), '\n')
    print(io, "├── constants: ", summary(model.constants), '\n')
    print(io, "├── σgrid: ", size(model.σgrid, 3), " σ levels", '\n')
    print(io, "├── stress_balance: ", summary(model.stress_balance), '\n')
    print(io, "├── rheology: ", summary(model.rheology), '\n')
    print(io, "├── friction: ", summary(model.friction), '\n')
    print(io, "├── grounding: ", summary(model.grounding), '\n')
    print(io, "├── masking: ", summary(model.masking), '\n')
    print(io, "├── velocity_solver: ", summary(model.velocity_solver), '\n')
    print(io, "├── advection: ", summary(model.advection), '\n')
    print(io, "└── timestepper: ", summary(model.timestepper))
    return nothing
end
