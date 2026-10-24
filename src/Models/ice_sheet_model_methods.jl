using Oceananigans: fields, prognostic_fields, initialize!
using Oceananigans.Fields: set!
using Oceananigans.TimeSteppers: Clock, update_state!, reset!
using Oceananigans.OutputWriters: checkpointer_address, default_included_properties
using Oceananigans.Simulations: iteration
using Oceananigans.Utils: prettytime, prettysummary

const ISM = IceSheetModel

"""
$(TYPEDSIGNATURES)

Set the ice thickness `H`, surface elevation `s` and depth-averaged velocities `u`, `v`.
"""
function Oceananigans.Fields.set!(model::ISM; H = nothing, s = nothing, u = nothing, v = nothing)
    set_if_given!(model.geometry.H, H)
    set_if_given!(model.geometry.s, s)
    set_if_given!(model.velocities.u, u)
    set_if_given!(model.velocities.v, v)
    update_state!(model)
    return nothing
end

set_if_given!(field, ::Nothing) = nothing
set_if_given!(field, value) = set!(field, value)

Oceananigans.Fields.set!(model::ISM, new_clock::Clock) = set!(model.clock, new_clock)

Oceananigans.Architectures.architecture(model::ISM) = model.architecture
Oceananigans.Simulations.iteration(model::ISM) = model.clock.iteration
Oceananigans.Utils.prettytime(model::ISM) = prettytime(model.clock.time)
Oceananigans.initialize!(::ISM) = nothing
Oceananigans.TimeSteppers.reset!(::ISM) = nothing
Oceananigans.OutputWriters.default_included_properties(::ISM) = [:grid]
Oceananigans.OutputWriters.checkpointer_address(::ISM) = "IceSheetModel"

Oceananigans.fields(model::ISM) = (; u = model.velocities.u, v = model.velocities.v, model.geometry...,
                                     b = model.bed, a = model.surface_mass_balance, m = model.basal_melt_rate)

Oceananigans.prognostic_fields(model::ISM) = (; H = model.geometry.H)

function Oceananigans.TimeSteppers.update_state!(model::ISM, callbacks = [])
    fill_halo_regions!(model.geometry.H, model.clock, fields(model))
    fill_halo_regions!((model.velocities.u, model.velocities.v), model.clock, fields(model))
    return nothing
end

function Oceananigans.prognostic_state(model::ISM)
    return (clock = prognostic_state(model.clock),
            geometry = prognostic_state(model.geometry),
            velocities = prognostic_state(model.velocities),
            timestepper = prognostic_state(model.timestepper))
end

function Oceananigans.restore_prognostic_state!(restored::ISM, from)
    restore_prognostic_state!(restored.clock, from.clock)
    restore_prognostic_state!(restored.geometry, from.geometry)
    restore_prognostic_state!(restored.velocities, from.velocities)
    restore_prognostic_state!(restored.timestepper, from.timestepper)
    return restored
end

Oceananigans.restore_prognostic_state!(::ISM, ::Nothing) = nothing

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
