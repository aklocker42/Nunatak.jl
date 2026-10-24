const ISM = IceSheetModel

"""
$(TYPEDSIGNATURES)

Set the ice thickness `H`, surface elevation `s` and depth-averaged velocities `u`, `v`.
"""
function set!(model::ISM; H = nothing, s = nothing, u = nothing, v = nothing)
    set_if_given!(model.geometry.H, H)
    set_if_given!(model.geometry.s, s)
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

function update_state!(model::ISM, callbacks = [])
    fill_halo_regions!(model.geometry.H, model.clock, fields(model))
    fill_halo_regions!((model.velocities.u, model.velocities.v), model.clock, fields(model))
    return nothing
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
    print(io, "├── σgrid: ", size(model.σgrid, 3), " σ levels", '\n')
    print(io, "├── stress_balance: ", summary(model.stress_balance), '\n')
    print(io, "├── rheology: ", summary(model.rheology), '\n')
    print(io, "├── friction: ", summary(model.friction), '\n')
    print(io, "├── grounding: ", summary(model.grounding), '\n')
    print(io, "└── timestepper: ", summary(model.timestepper))
    return nothing
end
