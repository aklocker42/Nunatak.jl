# Numerics

## Grids

The horizontal grid is an Oceananigans `RectilinearGrid` with a `Flat` vertical direction, on CPUs or GPUs. Fields
use the Arakawa C-grid. The thickness, bed and other scalars sit at cell centres, ``u`` on x-faces and ``v`` on
y-faces. Shear stresses sit at cell corners.

Column quantities live on the [`SigmaGrid`](@ref), a three-dimensional grid with the same horizontal coordinates
and `Nσ` levels in the normalized height above the bed, ``\zeta = 1 - \sigma \in [0, 1]``. The physical height is
``z = b + \zeta H``. `bed_refinement > 1` clusters the levels toward the bed, where the vertical shear and the
temperate layer sit.

## Velocity solve

The depth-integrated stress balance is nonlinear in the velocity through the viscosity and the friction law. A
[`StressBalanceSolver`](@ref) combines two iterations.

**Picard iteration** ([`PicardIteration`](@ref)): freeze the viscosity and drag at the current velocity, solve the
linear problem, repeat. The iteration is controlled by

- `Convergence(; reltol, abstol, maxiter)`: iterate until the residual falls below ``\max(\text{reltol}\,\|r_0\|, \text{abstol})``;
- `FixedIterations(N)`: iterate exactly `N` times. Within a time-dependent run, a few iterations per step lag the
  nonlinearity by one step, which is cheap and usually sufficient.

**Conjugate gradients** ([`VelocityConjugateGradientSolver`](@ref)) solve each linear problem. Each row of the
operator is integrated over its velocity cell, so the operator is symmetric positive definite on any
`RectilinearGrid`. The available preconditioners are

- [`JacobiPreconditioner`](@ref): a stored diagonal;
- [`StressBalanceDiagonalPreconditioner`](@ref) (the default): the operator diagonal evaluated from the current
  coefficients, so it follows the Picard iterations without storage;
- [`StressBalanceMultigridPreconditioner`](@ref): a geometric multigrid V-cycle. Each level halves the grid in
  `x`, `y` or both and rediscretizes the operator. The smoother is ℓ1-Jacobi on points ([`PointJacobi`](@ref), the
  default) or on lines in `y` ([`LineJacobi`](@ref)), so the V-cycle stays symmetric positive definite. On flowlines
  and other grids much longer in `x` than in `y`, `LineJacobi()` needs far fewer iterations.

On a MISMIP+ ice stream with a grounding line and a calving front (`benchmarks/multigrid_scaling.jl`, one SSA solve
from zero to a relative tolerance of ``10^{-8}``), the diagonal preconditioner needs twice the iterations each time
the grid is refined, while multigrid stays nearly flat:

| Δx | Grid | Diagonal: iterations, CPU / GPU time (s) | Multigrid: iterations, CPU / GPU time (s) |
|---|---|---|---|
| 8 km | 80 × 10 | 42, 0.001 / 0.02 | 14, 0.005 / 0.02 |
| 4 km | 160 × 20 | 91, 0.006 / 0.05 | 17, 0.013 / 0.04 |
| 2 km | 320 × 40 | 188, 0.04 / 0.10 | 19, 0.09 / 0.05 |
| 1 km | 640 × 80 | 392, 0.65 / 0.22 | 21, 0.44 / 0.09 |
| 0.5 km | 1280 × 160 | 817, 3.2 / 0.56 | 23, 1.4 / 0.17 |
| 0.25 km | 2560 × 320 | 1767, – / 1.1 | 26, – / 0.39 |

CPU times are on one thread, GPU times on one NVIDIA H200. A V-cycle costs several operator applications, so on a
CPU multigrid is faster only from about 50 000 cells; on a GPU it is faster from about 3000 cells.

The `schedule` of the solver (default `IterationInterval(1)`) can skip velocity solves between time steps.

## Thickness transport and time stepping

The thickness is advanced with a split Runge–Kutta scheme (`timestepper = :SplitRungeKutta3` by default). Each
stage clips ``H`` at zero. The default advection is fifth-order WENO with bounds ``0 \le H \le 100`` km, which
needs a horizontal halo of at least 4; `UpwindBiased(order = 1)` works with any halo. The finite upper bound keeps
the limiter's reverse-mode derivative finite.

Two functions give stable time-step bounds:

- `Nunatak.cell_advection_timescale(model)`: the shortest time for the depth-averaged flow to cross a cell, for an
  advective CFL condition;
- `Nunatak.cell_diffusion_timescale(model)`: the explicit diffusion limit of the shallow-ice flux (infinite for the
  depth-integrated balances); see [Stress balances](stress_balances.md#Shallow-ice-approximation).
