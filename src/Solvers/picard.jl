"""
$(TYPEDEF)

Fixed-point (Picard) iteration on the nonlinear stress balance: freeze the effective
viscosity and friction at the current velocity, solve the linear SPD problem, repeat.

$(TYPEDFIELDS)
"""
struct PicardIteration{C, W}
    "`Convergence` or `FixedIterations`"
    iteration_control :: C
    "velocity from the previous Picard iterate"
    previous_velocity :: W
end

"""
$(TYPEDSIGNATURES)

Build a `PicardIteration` on `grid`.
"""
PicardIteration(grid; iteration_control = FixedIterations(10)) =
    PicardIteration(iteration_control, VelocityVector(grid))

"""
$(TYPEDSIGNATURES)

Run Picard iterations: `update_coefficients!(x, args...)` freezes the coefficients at `x`,
then `solve!(x, linear_solver, b, args...)` solves the linearised problem. Returns the number
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
    change = typemax(control.reltol)

    while change > control.reltol && iteration < control.maxiter
        copyto!(x⁻, x)
        update_coefficients!(x, args...)
        solve!(x, linear_solver, b, args...)
        axpy!(-1, x, x⁻)
        Δ = sqrt(vector_dot!(w, x⁻, x⁻))
        X = sqrt(vector_dot!(w, x, x))
        change = ifelse(iszero(X), Δ, Δ / X)
        iteration += 1
    end

    return iteration
end
