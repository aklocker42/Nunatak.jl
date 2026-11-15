# Model overview

Nunatak evolves the thickness ``H(x, y, t)`` of an ice sheet or ice shelf on a two-dimensional Oceananigans grid
whose vertical direction is `Flat`. Column quantities such as the viscosity, the velocity profile and the enthalpy
live on a terrain-following σ grid built from the same horizontal grid (see [Numerics](numerics.md)).

## Building a model

```julia
using Oceananigans, Oceananigans.Units, Nunatak

year = years = 31556926seconds

grid = RectilinearGrid(CPU(); size = (64, 32), halo = (4, 4), x = (0, 640kilometers), y = (0, 80kilometers),
                       topology = (Bounded, Bounded, Flat))

model = IceSheetModel(grid;
                      rheology = GlenFlowLaw(grid; rate_factor = 6.3e-25),
                      friction = WeertmanFriction(grid; coefficient = 3e6),
                      stress_balance = DepthIntegrated(shear = DIVAShear()),
                      grounding = SubgridGroundedFraction(),
                      bed = (x, y) -> -150 - 1e-3 * x,
                      surface_mass_balance = 0.3 / year)

set!(model, H = (x, y) -> ifelse(x < 600kilometers, 500.0, 0.0))
simulation = Simulation(model; Δt = 1year, stop_time = 10years)
run!(simulation)
```

The keywords of [`IceSheetModel`](@ref) choose each component:

| Keyword | Options | Default |
|---|---|---|
| `stress_balance` | `ShallowIce()`, `DepthIntegrated()` (SSA), `DepthIntegrated(shear = DIVAShear())` (DIVA) | SSA |
| `rheology` | `GlenFlowLaw` with a constant, spatially varying or `ArrheniusRateFactor` rate factor | `nothing` (velocities stay fixed) |
| `friction` | `WeertmanFriction`, `RegularizedCoulombFriction`, or `nothing` (free slip) | `nothing` |
| `grounding` | `SubgridGroundedFraction()`, or `nothing` (all ice grounded) | `nothing` |
| `masking` | `SmoothIceMask(grid)`, `PartialCellFront(grid)` | `SmoothIceMask(grid)` |
| `thermodynamics` | `EnthalpyThermodynamics(grid; ...)`, or `nothing` (isothermal) | `nothing` |
| `velocity_solver` | `StressBalanceSolver`, `ShallowIceSolver`, or `nothing` | chosen from the stress balance |
| `advection` | any Oceananigans advection scheme | bounded fifth-order WENO; `Centered(order = 2)` with `ShallowIce()` |
| `constants` | `IceSheetConstants(grid; ice_density, ocean_density, gravitational_acceleration)` | MISMIP+ densities |
| `bed`, `surface_mass_balance`, `basal_melt_rate` | number, array, function `(x, y)` or `Field` | `0` |
| `Nσ`, `bed_refinement` | levels of the σ grid and their clustering toward the bed | `10`, `2` |

All parameters that can be fields, such as the bed, the friction coefficient, the rate factor and the mass
balance, are stored as `Field`s so that gradients with respect to them exist (see [Gradients](../gradients.md)).
Units are SI throughout: metres, seconds, pascals; rates are per second. `Oceananigans.Units` has no year, so the
examples define `year = years = 31556926seconds`, the year of the MISMIP+ and EISMINT specifications.

## Mass balance

The prognostic variable is the thickness, transported by the depth-averaged velocity ``\bar{\mathbf{u}}``:

```math
\frac{\partial H}{\partial t} + \nabla \cdot (\bar{\mathbf{u}} H) = a - (1 - \phi)\, m - m_g ,
```

with surface mass balance ``a``, sub-shelf melt ``m`` acting on the floating fraction ``1 - \phi`` of a cell
(``\phi`` is the grounded fraction, see [Grounding lines and calving fronts](grounding_and_fronts.md)), and the
grounded basal melt ``m_g`` when the thermodynamics are coupled to the mass balance. The thickness is clipped at
zero after each stage; the clipped amount is accumulated in `geometry.clipped_thickness` so that the volume budget
closes.

## Geometry

Sea level is at ``z = 0``. From ``H`` and the bed ``b`` the model diagnoses

- the surface ``s = \max\left(b + H, (1 - \rho_i/\rho_w) H\right)``: on the bed where the ice is grounded and in
  hydrostatic balance where it floats;
- the grounded fractions ``\phi``, ``\phi^u``, ``\phi^v`` at cell centres and velocity points;
- the effective pressure ``N = \max(\rho_i g H + \rho_w g \min(b, 0), 0)``, which vanishes at and beyond the grounding line;
- the ice mask ``\mu``, which removes ice-free cells from the stress balance.

## One time step

`IceSheetModel` works with Oceananigans' `Simulation`, `run!`, output writers, `Checkpointer` and
`TimeStepWizard`. Each step, `time_step!(model, Δt)` advances the thickness with a split Runge–Kutta scheme (three stages by default). After
the final stage a [`PartialCellFront`](@ref) advances the calving front. `update_state!` then

1. diagnoses the geometry from the new thickness,
2. solves the stress balance for the velocities, when the velocity solver's `schedule` actuates,
3. advances the enthalpy by one implicit column step, when thermodynamics are present.

The velocities used during a step are therefore those of the geometry at the start of the step.
