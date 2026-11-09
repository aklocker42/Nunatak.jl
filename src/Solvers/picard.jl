"""
$(TYPEDEF)

Fixed-point (Picard) iteration on the nonlinear stress balance: freeze the effective
viscosity and friction at the current velocity, solve the linear SPD problem, repeat.

With `Convergence`, reverse-mode AD differentiates the converged solution rather than the iterates:
the adjoint is the fixed point of the adjoint of one Picard step (Christianson 1994), iterated to
`adjoint_iteration_control`, so the tape holds the converged state only.

$(TYPEDFIELDS)
"""
struct PicardIteration{C, A, W}
    "`Convergence` or `FixedIterations`"
    iteration_control :: C
    "`Convergence` of the fixed-point adjoint iteration, used when `iteration_control` is a `Convergence`"
    adjoint_iteration_control :: A
    "velocity from the previous Picard iterate"
    previous_velocity :: W
end

"""
$(TYPEDSIGNATURES)

Build a `PicardIteration` on `grid`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> PicardIteration(grid; iteration_control = Convergence(; reltol = 1e-8)).adjoint_iteration_control
Convergence{Float64}(1.0e-8, 0.0, 1000)
```
"""
PicardIteration(grid; iteration_control = FixedIterations(10), adjoint_iteration_control = iteration_control) =
    PicardIteration(iteration_control, adjoint_iteration_control, VelocityVector(grid))

"""
$(TYPEDSIGNATURES)

Run Picard iterations: `update_coefficients!(x, args...)` freezes the coefficients at `x`,
then `solve!(x, linear_solver, b, args...)` solves the linearized problem. Returns the number
of Picard iterations taken.
"""
function picard_solve!(x, picard::PicardIteration{<:FixedIterations}, linear_solver, update_coefficients!, b, args...)
    for _ in 1:picard.iteration_control.iterations
        update_coefficients!(x, args...)
        solve!(x, linear_solver, b, args...)
    end
    return picard.iteration_control.iterations
end

function picard_solve!(x, picard::PicardIteration{<:Convergence}, linear_solver, update_coefficients!, b, args...)
    control = picard.iteration_control
    x⁻ = picard.previous_velocity
    w = linear_solver.dot_workspace
    iteration = 0
    Δ = typemax(control.reltol)
    X = zero(control.reltol)

    while Δ > max(control.reltol * X, control.abstol) && iteration < control.maxiter
        copyto!(x⁻, x)
        update_coefficients!(x, args...)
        solve!(x, linear_solver, b, args...)
        axpy!(-1, x, x⁻)
        Δ = sqrt(vector_dot!(w, x⁻, x⁻))
        X = sqrt(vector_dot!(w, x, x))
        iteration += 1
    end

    return iteration
end
