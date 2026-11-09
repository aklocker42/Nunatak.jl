"""
$(TYPEDEF)

Nonlinear stress-balance solver: Picard iterations around a preconditioned conjugate gradient
solve of `ssa_operator!`.

$(TYPEDFIELDS)
"""
mutable struct StressBalanceSolver{L, P, S}
    "linear solver for the frozen-coefficient problem"
    linear_solver :: L
    "`PicardIteration`"
    picard :: P
    "schedule of the velocity solves in `update_state!`; the velocities stay fixed in between"
    schedule :: S
    "Picard iterations taken by the last solve"
    iteration :: Int
end

"""
$(TYPEDSIGNATURES)

Build a `StressBalanceSolver` on `grid` with the Jacobi preconditioner `StressBalanceDiagonalPreconditioner`;
`preconditioner = StressBalanceMultigridPreconditioner(grid)` takes far fewer iterations on large grids.
The velocities are solved in `update_state!` at iterations where `schedule(model)` is true and held fixed
otherwise; `set!` always solves. An `IterationInterval` solves at every stage of its iterations.
`adjoint_iteration_control` stops the fixed-point adjoint of converged Picard iterations.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> solver = StressBalanceSolver(grid; picard_iteration_control = FixedIterations(5));

julia> solver.picard.iteration_control
FixedIterations(5)

julia> StressBalanceSolver(grid; schedule = IterationInterval(48)).schedule
IterationInterval(48, 0)
```
"""
function StressBalanceSolver(grid;
                             linear_iteration_control = Convergence(eltype(grid); reltol = 1e-8, maxiter = 10_000),
                             picard_iteration_control = Convergence(eltype(grid); reltol = 1e-6, maxiter = 100),
                             adjoint_iteration_control = picard_iteration_control,
                             preconditioner = StressBalanceDiagonalPreconditioner(),
                             schedule = IterationInterval(1))
    linear_solver = VelocityConjugateGradientSolver(ssa_operator!, grid; iteration_control = linear_iteration_control, preconditioner)
    picard = PicardIteration(grid; iteration_control = picard_iteration_control, adjoint_iteration_control)
    return StressBalanceSolver(linear_solver, picard, schedule, 0)
end

Base.summary(solver::StressBalanceSolver) =
    string("StressBalanceSolver(picard = ", solver.picard.iteration_control,
           ", linear = ", solver.linear_solver.iteration_control,
           ", schedule = ", summary(solver.schedule), ")")
