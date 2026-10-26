"""
$(TYPEDSIGNATURES)

Take `Nt` time steps of size `Δt`.
"""
function time_steps!(model, Δt, Nt)
    for _ in 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

"""
    checkpointed_gradient!(dmodel, cost, model, Δt, Nt; checkpoint_interval = ceil(Int, sqrt(Nt)))

Gradient of `cost(model)` after `Nt` time steps of size `Δt` with respect to the initial state and the
parameter fields, accumulated into the shadow `dmodel` (from `Enzyme.make_zero(model)`), and the value
of the cost. The forward run stores `prognostic_state` every `checkpoint_interval` steps; the reverse
pass restores each checkpoint with `restore_prognostic_state!` and differentiates its segment, so only
one segment is taped at a time. `model` is left at the final state. Requires Enzyme.
"""
function checkpointed_gradient! end
