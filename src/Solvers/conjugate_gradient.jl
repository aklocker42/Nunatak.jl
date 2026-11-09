"""
$(TYPEDEF)

Diagonal (Jacobi) preconditioner: `z = r / diagonal`.

$(TYPEDFIELDS)
"""
struct JacobiPreconditioner{D}
    "operator diagonal as a `VelocityVector`"
    diagonal :: D
end

"""
$(TYPEDSIGNATURES)

Apply the preconditioner `P` to the residual `r`, storing the result in `z`.
"""
precondition!(z, ::Nothing, r, args...) = copyto!(z, r)

# Preconditioners that store state built from the operator coefficients refresh it once per solve.
update_preconditioner!(P, args...) = nothing

function precondition!(z::VelocityVector, P::JacobiPreconditioner, r::VelocityVector, args...)
    return divide!(z, r, P.diagonal)
end

"""
$(TYPEDEF)

Preconditioned conjugate gradient solver for a symmetric positive-definite operator acting on
a `VelocityVector`. The operator follows the Oceananigans convention
`linear_operation!(q, p, args...)`; `adjoint_operation!` defaults to `linear_operation!`
because the Picard (SSA/DIVA) operator is self-adjoint.

$(TYPEDFIELDS)
"""
mutable struct VelocityConjugateGradientSolver{L, AL, C, P, W}
    "forward operator `linear_operation!(q, p, args...)`"
    linear_operation! :: L
    "adjoint operator `adjoint_operation!(q, p, args...)`"
    adjoint_operation! :: AL
    "`Convergence` or `FixedIterations`"
    iteration_control :: C
    "preconditioner, `nothing` or `JacobiPreconditioner`"
    preconditioner :: P
    "residual"
    residual :: W
    "preconditioned residual"
    preconditioned_residual :: W
    "search direction"
    search_direction :: W
    "operator applied to the search direction"
    operator_product :: W
    "workspace for elementwise products in dot products"
    dot_workspace :: W
    "iterations taken by the last solve"
    iteration :: Int
end

"""
$(TYPEDSIGNATURES)

Build a `VelocityConjugateGradientSolver` on `grid`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> identity!(q, p) = copyto!(q, p);

julia> solver = VelocityConjugateGradientSolver(identity!, grid; iteration_control = FixedIterations(2));

julia> solver.iteration_control
FixedIterations(2)
```
"""
function VelocityConjugateGradientSolver(linear_operation!, grid;
                                         adjoint_operation! = linear_operation!,
                                         iteration_control = Convergence(eltype(grid)),
                                         preconditioner = nothing)
    template = VelocityVector(grid)
    return VelocityConjugateGradientSolver(linear_operation!, adjoint_operation!,
                                           iteration_control, preconditioner,
                                           template, VelocityVector(grid), VelocityVector(grid), VelocityVector(grid), VelocityVector(grid),
                                           0)
end

"""
$(TYPEDSIGNATURES)

Solve `A x = b` in place, starting from the initial guess in `x`.
"""
solve!(x, solver::VelocityConjugateGradientSolver, b, args...) =
    conjugate_gradient!(x, solver.linear_operation!, solver, b, args...)

"""
$(TYPEDSIGNATURES)

Solve `Aᵀ λ = b` in place, starting from the initial guess in `λ`.
"""
adjoint_solve!(λ, solver::VelocityConjugateGradientSolver, b, args...) =
    conjugate_gradient!(λ, solver.adjoint_operation!, solver, b, args...)

function conjugate_gradient!(x, operation!, solver, b, args...)
    r = solver.residual
    z = solver.preconditioned_residual
    p = solver.search_direction
    q = solver.operator_product
    w = solver.dot_workspace

    fill_halo_regions!(x)
    operation!(q, x, args...)
    subtract!(r, b, q)
    update_preconditioner!(solver.preconditioner, args...)
    precondition!(z, solver.preconditioner, r, args...)
    copyto!(p, z)
    ρ = vector_dot!(w, r, z)

    solver.iteration = iterate_conjugate_gradient!(x, operation!, solver, solver.iteration_control, ρ, args...)

    return x
end

function iterate_conjugate_gradient!(x, operation!, solver, control::FixedIterations, ρ, args...)
    for _ in 1:control.iterations
        ρ = conjugate_gradient_step!(x, operation!, solver, ρ, args...)
    end
    return control.iterations
end

function iterate_conjugate_gradient!(x, operation!, solver, control::Convergence, ρ, args...)
    w = solver.dot_workspace
    r = solver.residual
    r₀ = sqrt(vector_dot!(w, r, r))
    tolerance = max(control.reltol * r₀, control.abstol)
    iteration = 0
    residual_norm = r₀

    while residual_norm > tolerance && iteration < control.maxiter
        ρ = conjugate_gradient_step!(x, operation!, solver, ρ, args...)
        residual_norm = sqrt(vector_dot!(w, r, r))
        iteration += 1
    end

    return iteration
end

function conjugate_gradient_step!(x, operation!, solver, ρ, args...)
    r = solver.residual
    z = solver.preconditioned_residual
    p = solver.search_direction
    q = solver.operator_product
    w = solver.dot_workspace

    fill_halo_regions!(p)
    operation!(q, p, args...)
    pq = vector_dot!(w, p, q)
    α = ifelse(iszero(pq), zero(ρ), ρ / pq)

    axpy!(α, p, x)
    axpy!(-α, q, r)

    precondition!(z, solver.preconditioner, r, args...)
    ρ⁺ = vector_dot!(w, r, z)
    β = ifelse(iszero(ρ), zero(ρ), ρ⁺ / ρ)
    xpby!(z, β, p)

    return ρ⁺
end

Base.summary(solver::VelocityConjugateGradientSolver) =
    string("VelocityConjugateGradientSolver(", solver.iteration_control, ")")
