"""
$(TYPEDEF)

Nonlinear stress-balance solver: Picard iterations around a preconditioned conjugate gradient
solve of `ssa_operator!`.

$(TYPEDFIELDS)
"""
mutable struct StressBalanceSolver{L, P}
    "linear solver for the frozen-coefficient problem"
    linear_solver :: L
    "`PicardIteration`"
    picard :: P
    "Picard iterations taken by the last solve"
    iteration :: Int
end

"""
$(TYPEDSIGNATURES)

Build a `StressBalanceSolver` on `grid` with the Jacobi preconditioner `StressBalanceDiagonalPreconditioner`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> solver = StressBalanceSolver(grid; picard_iteration_control = FixedIterations(5));

julia> solver.picard.iteration_control
FixedIterations(5)
```
"""
function StressBalanceSolver(grid;
                             linear_iteration_control = Convergence(eltype(grid); reltol = 1e-8, maxiter = 10_000),
                             picard_iteration_control = Convergence(eltype(grid); reltol = 1e-6, maxiter = 100),
                             preconditioner = StressBalanceDiagonalPreconditioner())
    linear_solver = VelocityConjugateGradientSolver(ssa_operator!, grid; iteration_control = linear_iteration_control, preconditioner)
    picard = PicardIteration(grid; iteration_control = picard_iteration_control)
    return StressBalanceSolver(linear_solver, picard, 0)
end

Base.summary(solver::StressBalanceSolver) =
    string("StressBalanceSolver(picard = ", solver.picard.iteration_control,
           ", linear = ", solver.linear_solver.iteration_control, ")")

default_velocity_solver(grid, ::Nothing) = nothing
default_velocity_solver(grid, rheology) = StressBalanceSolver(grid)
