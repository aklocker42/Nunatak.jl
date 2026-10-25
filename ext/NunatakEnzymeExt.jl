module NunatakEnzymeExt

using Nunatak
using Nunatak: VelocityVector, VelocityConjugateGradientSolver, axpy!
using Nunatak.Oceananigans: Field
using Nunatak.Oceananigans.BoundaryConditions: fill_halo_regions!
using Enzyme
using Enzyme.EnzymeRules

import Nunatak: solve!

# Implicit adjoint of x = A(p)⁻¹ b: λ = A⁻ᵀ x̄, b̄ += λ, p̄ -= (∂(A x)/∂p)ᵀ λ.

const Shadowed = Union{Duplicated, MixedDuplicated}

shadow(a::Duplicated) = a.dval
shadow(a::MixedDuplicated) = a.dval[]

duplicated(value, ::Const) = Const(value)
duplicated(value, a::Shadowed) = Duplicated(value, shadow(a))

function copy_vector(x::VelocityVector)
    y = similar(x)
    parent(y.u) .= parent(x.u)
    parent(y.v) .= parent(x.v)
    return y
end

# Later Picard iterations overwrite the coefficients, so the reverse pass needs their values at solve time.
function copy_argument(a::Field)
    c = similar(a)
    parent(c) .= parent(a)
    return c
end

copy_argument(a) = a

function EnzymeRules.augmented_primal(config, func::Const{typeof(solve!)}, ::Type{RT},
                                      x::Shadowed,
                                      solver::Const{<:VelocityConjugateGradientSolver},
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
                             solver::Const{<:VelocityConjugateGradientSolver},
                             b, args...) where RT

    x̄ = shadow(x)
    solution, vals = tape

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

    fill!(x̄, 0)

    return (nothing, nothing, nothing, map(a -> nothing, args)...)
end

accumulate_right_hand_side_adjoint!(::Const, λ) = nothing
accumulate_right_hand_side_adjoint!(b::Shadowed, λ) = axpy!(1, λ, shadow(b))

end # module
