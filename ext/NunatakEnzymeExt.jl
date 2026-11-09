module NunatakEnzymeExt

using Nunatak
using Nunatak: VelocityVector, VelocityConjugateGradientSolver, PicardIteration, Convergence
using Nunatak: axpy!, subtract!, vector_dot!, time_steps!
using Nunatak: update_stress_balance_coefficients!, update_diva_coefficients!
using Nunatak: GlenFlowLaw, ArrheniusRateFactor
using Nunatak.Oceananigans: prognostic_state, restore_prognostic_state!, time_step!
using Nunatak.Oceananigans: Field
using Nunatak.Oceananigans.BoundaryConditions: fill_halo_regions!
using Enzyme
using Enzyme.EnzymeRules

import Nunatak: solve!, picard_solve!, checkpointed_gradient!

# Implicit adjoint of x = A(p)⁻¹ b: λ = A⁻ᵀ x̄, b̄ += λ, p̄ -= (∂(A x)/∂p)ᵀ λ.

const Shadowed = Union{Duplicated, MixedDuplicated}

shadow(a::Duplicated) = a.dval
shadow(a::MixedDuplicated) = a.dval[]

duplicated(value, ::Const) = Const(value)
duplicated(value, a::Shadowed) = Duplicated(value, shadow(a))

# Inside a differentiated model the solver is shadowed too; its workspace carries no gradient.
const Solver = Union{Const{<:VelocityConjugateGradientSolver}, Duplicated{<:VelocityConjugateGradientSolver},
                     MixedDuplicated{<:VelocityConjugateGradientSolver}}

function copy_vector(x::VelocityVector)
    y = similar(x)
    parent(y.u) .= parent(x.u)
    parent(y.v) .= parent(x.v)
    return y
end

# Enzyme can hand over a shadow whose data arrays have lost their offsets, so shadows are only accessed through `parent`.
function adjoint_seed(x̄, solution)
    y = similar(solution)
    parent(y.u) .= parent(x̄.u)
    parent(y.v) .= parent(x̄.v)
    return y
end

function zero_shadow!(x̄::VelocityVector)
    parent(x̄.u) .= 0
    parent(x̄.v) .= 0
    return nothing
end

# The cells the solver updates; outer wall faces of `Bounded` grids are boundary values.
solved_cells(f) = view(parent(f), f.grid.Hx .+ (1:f.grid.Nx), f.grid.Hy .+ (1:f.grid.Ny), :)

# Later Picard iterations overwrite the coefficients, so the reverse pass needs their values at solve time.
function copy_argument(a::Field)
    c = similar(a)
    parent(c) .= parent(a)
    return c
end

copy_argument(a) = a

function EnzymeRules.augmented_primal(config, func::Const{typeof(solve!)}, ::Type{RT},
                                      x::Shadowed,
                                      solver::Solver,
                                      b, args...) where RT

    solve!(x.val, solver.val, b.val, map(a -> a.val, args)...)
    # The solver updates interiors only; ∂(A x)/∂p needs the halos of the solution.
    fill_halo_regions!(x.val)
    tape = (copy_vector(x.val), map(a -> copy_argument(a.val), args))
    primal = EnzymeRules.needs_primal(config) ? x.val : nothing
    x̄ = EnzymeRules.needs_shadow(config) ? shadow(x) : nothing

    return EnzymeRules.AugmentedReturn(primal, x̄, tape)
end

function EnzymeRules.reverse(config, func::Const{typeof(solve!)}, ::Type{RT}, tape,
                             x::Shadowed,
                             solver::Solver,
                             b, args...) where RT

    solution, vals = tape
    x̄ = adjoint_seed(shadow(x), solution)

    λ = similar(x̄)
    fill!(λ, 0)
    adjoint_solve!(λ, solver.val, x̄, vals...)

    accumulate_right_hand_side_adjoint!(b, λ)

    q = similar(x̄)
    q̄ = similar(x̄)
    fill!(q̄, 0)
    axpy!(-1, λ, q̄)
    Enzyme.autodiff(set_runtime_activity(Reverse), Const(solver.val.linear_operation!), Const,
                    Duplicated(q, q̄), Const(solution), map(duplicated, vals, args)...)

    zero_shadow!(shadow(x))

    return (nothing, nothing, nothing, map(a -> nothing, args)...)
end

accumulate_right_hand_side_adjoint!(::Const, λ) = nothing
function accumulate_right_hand_side_adjoint!(b::Shadowed, λ)
    b̄ = shadow(b)
    solved_cells(b̄.u) .+= solved_cells(λ.u)
    solved_cells(b̄.v) .+= solved_cells(λ.v)
    return nothing
end

#####
##### Fixed-point adjoint of converged Picard iterations (Christianson 1994)
#####

# One Picard step maps the state z = (x, s) to (A(c)⁻¹ b, s′) with c, s′ = update(x, s), where s holds
# the coefficients the update overwrites (DIVA reads its previous βᵇ and F₂). At the fixed point the
# adjoint w = z̄ + (∂z′/∂z)ᵀ w is iterated with λ = A⁻ᵀ w_x, then p̄ = (∂z′/∂p)ᵀ w and b̄ = λ.

const ConvergedPicard = Enzyme.EnzymeCore.Annotation{<:PicardIteration{<:Convergence}}

# Positions in `args` of the coefficients the update overwrites; without friction the drag is left untouched.
written_coefficients(::typeof(update_stress_balance_coefficients!), η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, args...) = (1, 2, 3, 4, 5, 6)
written_coefficients(::typeof(update_stress_balance_coefficients!), η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, ::Nothing, args...) = (1, 2, 5, 6)
written_coefficients(::typeof(update_diva_coefficients!), η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, args...) = (1, 2, 3, 4, 5, 6, 14, 15, 16, 17)
written_coefficients(::typeof(update_diva_coefficients!), η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, ::Nothing, args...) = (1, 2, 3, 4, 5, 6, 16, 17)

shadowed_indices(i, ::Const) = ()
shadowed_indices(i, ::Shadowed) = (i,)

state_indices(written, args) = foldl((indices, i) -> (indices..., shadowed_indices(i, args[i])...), written; init = ())

# The reverse pass re-evaluates the coefficient update, so it needs the fields as they were at convergence.
snapshot(a::Field) = copy_argument(a)
snapshot(a) = a
# Later time steps in the taped segment overwrite the Arrhenius rate factor.
snapshot(r::GlenFlowLaw) = GlenFlowLaw(snapshot(r.rate_factor), r.exponent, r.strain_rate_regularization)
snapshot(r::ArrheniusRateFactor) = ArrheniusRateFactor(r.parameters, snapshot(r.field))

function zeroed(x::VelocityVector)
    y = similar(x)
    fill!(y, 0)
    return y
end

scratch_shadow(value, ::Const) = Const(value)
scratch_shadow(value, ::Shadowed) = Duplicated(value, Enzyme.make_zero(value))

reset_shadow!(::Const) = nothing
reset_shadow!(a::Duplicated) = Enzyme.remake_zero!(a.dval)

# The coefficient update of one Picard step, then A(c) x with x frozen at the fixed point.
function picard_coefficient_step!(q, x, xᵖ, operation!, update_coefficients!, args...)
    update_coefficients!(xᵖ, args...)
    operation!(q, x, args...)
    return nothing
end

function picard_step_adjoint!(ȳ, q̄, λ, solution, operation!, update!, shadows)
    q = similar(ȳ)
    xᵖ = copy_vector(solution)
    fill!(q̄, 0)
    axpy!(-1, λ, q̄)
    fill!(ȳ, 0)
    Enzyme.autodiff(set_runtime_activity(Reverse), Const(picard_coefficient_step!), Const,
                    Duplicated(q, q̄), Const(solution), Duplicated(xᵖ, ȳ), Const(operation!), Const(update!), shadows...)
    return nothing
end

function EnzymeRules.augmented_primal(config, func::Const{typeof(picard_solve!)}, ::Type{RT},
                                      x::Shadowed,
                                      picard::ConvergedPicard,
                                      linear_solver::Solver,
                                      update_coefficients!::Const,
                                      b, args...) where RT

    vals = map(a -> a.val, args)
    iterations = picard_solve!(x.val, picard.val, linear_solver.val, update_coefficients!.val, b.val, vals...)
    fill_halo_regions!(x.val)
    tape = (copy_vector(x.val), map(snapshot, vals))
    primal = EnzymeRules.needs_primal(config) ? iterations : nothing

    return EnzymeRules.AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.reverse(config, func::Const{typeof(picard_solve!)}, ::Type{RT}, tape,
                             x::Shadowed,
                             picard::ConvergedPicard,
                             linear_solver::Solver,
                             update_coefficients!::Const,
                             b, args...) where RT

    solution, vals = tape
    solver = linear_solver.val
    operation! = solver.linear_operation!
    update! = update_coefficients!.val
    x̄ = adjoint_seed(shadow(x), solution)

    state = state_indices(written_coefficients(update!, vals...), args)
    s̄ = map(i -> copy(parent(shadow(args[i]))), state)
    sweep = map(zero, s̄)

    λ = zeroed(x̄)
    λ⁺ = zeroed(x̄)
    ȳ = similar(x̄)
    q̄ = similar(x̄)
    r = similar(x̄)
    scratch = map(scratch_shadow, vals, args)
    control = picard.val.adjoint_iteration_control
    w = solver.dot_workspace
    iteration = 0
    Δ = typemax(control.reltol)
    M = zero(control.reltol)

    while Δ > max(control.reltol * M, control.abstol) && iteration < control.maxiter
        foreach(reset_shadow!, scratch)
        for (n, i) in enumerate(state)
            parent(scratch[i].dval) .= s̄[n] .+ sweep[n]
        end
        picard_step_adjoint!(ȳ, q̄, λ, solution, operation!, update!, scratch)
        for (n, i) in enumerate(state)
            sweep[n] .= parent(scratch[i].dval)
        end
        copyto!(r, x̄)
        axpy!(1, ȳ, r)
        copyto!(λ⁺, λ)
        adjoint_solve!(λ⁺, solver, r, vals...)
        subtract!(r, λ⁺, λ)
        copyto!(λ, λ⁺)
        Δ = sqrt(vector_dot!(w, r, r))
        M = sqrt(vector_dot!(w, λ, λ))
        iteration += 1
    end

    Δ > max(control.reltol * M, control.abstol) && @warn "Picard adjoint did not converge in $(control.maxiter) iterations: ‖Δλ‖ = $Δ, ‖λ‖ = $M"

    accumulate_right_hand_side_adjoint!(b, λ)

    for (n, i) in enumerate(state)
        parent(shadow(args[i])) .= s̄[n] .+ sweep[n]
    end
    picard_step_adjoint!(ȳ, q̄, λ, solution, operation!, update!, map(duplicated, vals, args))

    # A converged fixed point does not depend on the initial guess of x or of the lagged coefficients.
    for i in state
        parent(shadow(args[i])) .= 0
    end
    zero_shadow!(shadow(x))

    return (nothing, nothing, nothing, nothing, nothing, map(a -> nothing, args)...)
end

# prognostic_state aliases the fields on the CPU, so checkpoints are copies.
checkpoint(model) = deepcopy(prognostic_state(model))

function checkpointed_gradient!(dmodel, cost, model, Δt, Nt; checkpoint_interval = ceil(Int, sqrt(Nt)))
    checkpoints = map(1:checkpoint_interval:Nt) do n
        state = checkpoint(model)
        time_steps!(model, Δt, min(checkpoint_interval, Nt - n + 1))
        return state
    end
    final_state = checkpoint(model)

    _, J = autodiff(set_runtime_activity(ReverseWithPrimal), cost, Active, Duplicated(model, dmodel))

    for (s, n) in Iterators.reverse(enumerate(1:checkpoint_interval:Nt))
        restore_prognostic_state!(model, checkpoints[s])
        autodiff(set_runtime_activity(Reverse), time_steps!, Const, Duplicated(model, dmodel),
                 Const(Δt), Const(min(checkpoint_interval, Nt - n + 1)))
    end

    restore_prognostic_state!(model, final_state)
    return J
end

end # module
