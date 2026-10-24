module NunatakEnzymeExt

using Nunatak
using Nunatak: VelocityVector, VelocityConjugateGradientSolver, axpy!
using Enzyme
using Enzyme.EnzymeRules

import Nunatak: solve!

# Implicit adjoint of x = A(p)⁻¹ b: λ = A⁻ᵀ x̄, b̄ += λ, p̄ -= (∂(A x)/∂p)ᵀ λ.

const Shadowed = Union{Duplicated, MixedDuplicated}

shadow(a::Duplicated) = a.dval
shadow(a::MixedDuplicated) = a.dval[]

duplicated(a::Const) = a
duplicated(a::Shadowed) = Duplicated(a.val, shadow(a))

function copy_vector(x::VelocityVector)
    y = similar(x)
    parent(y.u) .= parent(x.u)
    parent(y.v) .= parent(x.v)
    return y
end

function EnzymeRules.augmented_primal(config, func::Const{typeof(solve!)}, ::Type{RT},
                                      x::Shadowed,
                                      solver::Const{<:VelocityConjugateGradientSolver},
                                      b, args...) where RT

    solve!(x.val, solver.val, b.val, map(a -> a.val, args)...)
    tape = copy_vector(x.val)
    primal = EnzymeRules.needs_primal(config) ? x.val : nothing
    x̄ = EnzymeRules.needs_shadow(config) ? shadow(x) : nothing

    return EnzymeRules.AugmentedReturn(primal, x̄, tape)
end

function EnzymeRules.reverse(config, func::Const{typeof(solve!)}, ::Type{RT}, tape,
                             x::Shadowed,
                             solver::Const{<:VelocityConjugateGradientSolver},
                             b, args...) where RT

    x̄ = shadow(x)
    vals = map(a -> a.val, args)

    λ = similar(x̄)
    fill!(λ, 0)
    adjoint_solve!(λ, solver.val, x̄, vals...)

    accumulate_right_hand_side_adjoint!(b, λ)

    q = similar(x̄)
    q̄ = similar(x̄)
    fill!(q̄, 0)
    axpy!(-1, λ, q̄)
    Enzyme.autodiff(set_runtime_activity(Reverse), Const(solver.val.linear_operation!), Const,
                    Duplicated(q, q̄), Const(tape), map(duplicated, args)...)

    fill!(x̄, 0)

    return (nothing, nothing, nothing, map(a -> nothing, args)...)
end

accumulate_right_hand_side_adjoint!(::Const, λ) = nothing
accumulate_right_hand_side_adjoint!(b::Shadowed, λ) = axpy!(1, λ, shadow(b))

end # module
