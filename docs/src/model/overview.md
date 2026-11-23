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
- the effective pressure ``N = \max(\rho_i g H + \rho_w g \min(b, 0), 0)``, which vanishes at and beyond the grounding line,
  or the till effective pressure from the basal water with `TillEffectivePressure`;
- the ice mask ``\mu``, which removes ice-free cells from the stress balance.

## One time step

`IceSheetModel` works with Oceananigans' `Simulation`, `run!`, output writers, `Checkpointer` and
`TimeStepWizard`. Each step, `time_step!(model, Δt)` advances the thickness with a split Runge–Kutta scheme (three stages by default).
Every stage restarts from the thickness at the start of the step, and in the final stage a [`PartialCellFront`](@ref) advances
the calving front. After every stage `update_state!`

1. diagnoses the geometry from the new thickness,
2. solves the stress balance for the velocities, when the velocity solver's `schedule` actuates,
3. advances the enthalpy by one implicit column step, in the final stage only, when thermodynamics are present.

Each stage therefore uses the velocities of the previous stage's thickness, and the first stage those left by the previous step.

```@raw html
<div><svg class="nunatak-diagram" viewBox="0 0 720 742" role="img" aria-label="Call sequence of one IceSheetModel time step" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="nd-head-step" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="nd-head" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<rect class="nd-key" x="10" y="10" width="700" height="40" rx="6"/>
<text class="nd-mono" x="360.0" y="26.5" text-anchor="middle">time_step!(model, Δt)</text>
<text class="nd-note" x="360.0" y="42.5" text-anchor="middle">Oceananigans split Runge–Kutta, three stages by default</text>
<line class="nd-arrow" x1="360" y1="50" x2="360" y2="64" marker-end="url(#nd-head-step)"/>
<rect class="nd-box" x="10" y="64" width="700" height="28" rx="6"/>
<text class="nd-mono" x="360.0" y="82.5" text-anchor="middle">cache_current_fields!   H⁻ = H</text>
<line class="nd-arrow" x1="360" y1="92" x2="360" y2="112" marker-end="url(#nd-head-step)"/>
<rect class="nd-frame" x="10" y="112" width="700" height="580" rx="8"/>
<rect class="nd-tab" x="22" y="102" width="196.0" height="20" rx="4"/>
<text class="nd-tabtext" x="30" y="116">for stage = 1, …, Nstages</text>
<rect class="nd-box" x="30" y="132" width="660" height="62" rx="6"/>
<text class="nd-mono" x="360.0" y="151.5" text-anchor="middle">rk_substep!</text>
<text class="nd-mono" x="360.0" y="167.5" text-anchor="middle">compute_thickness_tendency!   G = a − m − mᵍ − ∇⋅(ūH)</text>
<text class="nd-mono" x="360.0" y="183.5" text-anchor="middle">H = max(H⁻ + Δτ G, 0)      final stage, PartialCellFront: advance_front!</text>
<line class="nd-arrow" x1="360" y1="194" x2="360" y2="206" marker-end="url(#nd-head-step)"/>
<rect class="nd-box" x="30" y="206" width="660" height="26" rx="6"/>
<text class="nd-note" x="360.0" y="223.5" text-anchor="middle">clock moves to the stage time</text>
<line class="nd-arrow" x1="360" y1="232" x2="360" y2="258" marker-end="url(#nd-head-step)"/>
<rect class="nd-subframe" x="30" y="258" width="660" height="418" rx="8"/>
<rect class="nd-tab" x="42" y="248" width="109.60000000000001" height="20" rx="4"/>
<text class="nd-tabtext" x="50" y="262">update_state!</text>
<rect class="nd-box" x="50" y="276" width="620" height="42" rx="6"/>
<text class="nd-mono" x="360.0" y="293.5" text-anchor="middle">update_geometry_state!   →   update_geometry!</text>
<text class="nd-note" x="360.0" y="309.5" text-anchor="middle">surface s, grounded fractions φ, φᵘ, φᵛ, effective pressure N, ice mask</text>
<line class="nd-arrow" x1="360" y1="318" x2="360" y2="332" marker-end="url(#nd-head-step)"/>
<rect class="nd-box" x="50" y="332" width="620" height="42" rx="6"/>
<text class="nd-mono" x="360.0" y="349.5" text-anchor="middle">maybe_update_velocities!   →   update_velocities!(model, solver, stress_balance)</text>
<text class="nd-note" x="360.0" y="365.5" text-anchor="middle">when solver.schedule(model) actuates; dispatch on the stress balance</text>
<line class="nd-arrow" x1="360" y1="374" x2="140.0" y2="390" marker-end="url(#nd-head-step)"/>
<rect class="nd-branch" x="50" y="390" width="180" height="86" rx="6"/>
<text class="nd-note" x="140.0" y="413.5" text-anchor="middle">SSA</text>
<text class="nd-mono" x="140.0" y="429.5" text-anchor="middle">compute_driving_stress!</text>
<text class="nd-mono" x="140.0" y="445.5" text-anchor="middle">picard_solve!</text>
<text class="nd-mono" x="140.0" y="461.5" text-anchor="middle">extend_depth_average!</text>
<line class="nd-arrow" x1="360" y1="374" x2="336.0" y2="390" marker-end="url(#nd-head-step)"/>
<rect class="nd-branch" x="242" y="390" width="188" height="86" rx="6"/>
<text class="nd-note" x="336.0" y="413.5" text-anchor="middle">DIVA</text>
<text class="nd-mono" x="336.0" y="429.5" text-anchor="middle">compute_driving_stress!</text>
<text class="nd-mono" x="336.0" y="445.5" text-anchor="middle">picard_solve!</text>
<text class="nd-mono" x="336.0" y="461.5" text-anchor="middle">compute_velocity_profile!</text>
<line class="nd-arrow" x1="360" y1="374" x2="556.0" y2="390" marker-end="url(#nd-head-step)"/>
<rect class="nd-branch" x="442" y="390" width="228" height="86" rx="6"/>
<text class="nd-note" x="556.0" y="421.5" text-anchor="middle">ShallowIce</text>
<text class="nd-mono" x="556.0" y="437.5" text-anchor="middle">compute_shallow_ice_velocities!</text>
<text class="nd-mono" x="556.0" y="453.5" text-anchor="middle">compute_basal_drag!</text>
<line class="nd-arrow" x1="140" y1="476" x2="140" y2="490" marker-end="url(#nd-head-step)"/>
<line class="nd-arrow" x1="336" y1="476" x2="336" y2="490" marker-end="url(#nd-head-step)"/>
<rect class="nd-key" x="50" y="490" width="380" height="70" rx="6"/>
<text class="nd-mono" x="240.0" y="505.5" text-anchor="middle">picard_solve!   repeat until converged or N times</text>
<text class="nd-mono" x="240.0" y="521.5" text-anchor="middle">update_*_coefficients!   ησ, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ</text>
<text class="nd-mono" x="240.0" y="537.5" text-anchor="middle">solve!   conjugate gradients on ssa_operator!</text>
<text class="nd-note" x="240.0" y="553.5" text-anchor="middle">preconditioner: Jacobi or multigrid V-cycle</text>
<line class="nd-arrow" x1="360" y1="560" x2="360" y2="578" marker-end="url(#nd-head-step)"/>
<rect class="nd-box" x="50" y="578" width="620" height="26" rx="6"/>
<text class="nd-mono" x="360.0" y="595.5" text-anchor="middle">fill_velocity_halos!</text>
<line class="nd-arrow" x1="360" y1="604" x2="360" y2="618" marker-end="url(#nd-head-step)"/>
<rect class="nd-box" x="50" y="618" width="620" height="42" rx="6"/>
<text class="nd-mono" x="360.0" y="635.5" text-anchor="middle">maybe_step_thermodynamics!   →   compute_strain_heating!   →   advance_enthalpy!</text>
<text class="nd-note" x="360.0" y="651.5" text-anchor="middle">final stage only, when thermodynamics.schedule(model) actuates</text>
<line class="nd-arrow" x1="360" y1="692" x2="360" y2="706" marker-end="url(#nd-head-step)"/>
<rect class="nd-box" x="10" y="706" width="700" height="28" rx="6"/>
<text class="nd-mono" x="360.0" y="724.5" text-anchor="middle">clock.iteration += 1</text>
</svg></div>
```
