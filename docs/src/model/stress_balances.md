# Stress balances

Three approximations of the Stokes equations are available. The depth-integrated ones (SSA and DIVA) share a single
velocity operator and solver; the shallow-ice approximation needs no linear solve.

## Shallow-ice approximation

With `stress_balance = ShallowIce()` the velocity follows locally from the surface slope. In normalized height
above the bed ``\zeta = (z - b)/H``,

```math
u(\zeta) = u_b - 2 (\rho_i g)^n H^{n+1} |\nabla s|^{n-1} \frac{\partial s}{\partial x}
           \int_0^\zeta A(\zeta') (1 - \zeta')^n \, d\zeta' ,
```

and likewise for ``v``. The velocity is evaluated at the velocity points with the face-averaged thickness and rate
factor and the slope across the face averaged from the four neighbouring differences. With a
[`WeertmanFriction`](@ref) the sliding velocity balances the driving stress, ``C |u_b|^m = \rho_i g H |\nabla s|``;
without friction, and where ``C = 0``, the bed is frozen. Floating ice is held still.

The default thickness advection with `ShallowIce()` is `Centered(order = 2)`, so the thickness flux is the
diffusive flux ``-D \nabla s`` of [Mahaffy1976](@citet). Explicit time stepping of this flux is stable only for
``\Delta t`` well below the diffusive limit; `Nunatak.cell_diffusion_timescale(model)` returns the per-direction
bound ``\Delta^2 / D``. In two dimensions the linearized shallow-ice diffusivity exceeds ``D`` and both directions
add up, so use a `diffusive_cfl` well below 1/2 (the Halfar benchmark uses 1/4).

## Shallow-shelf approximation (SSA)

`stress_balance = DepthIntegrated()` solves for the depth-averaged velocity ``\bar{\mathbf{u}}`` from the balance of
membrane stresses, basal drag and driving stress:

```math
-\nabla \cdot \mathbf{T} + \beta \bar{\mathbf{u}} = -\rho_i g H \nabla s ,
```

with

```math
T^{xx} = 2 \bar\eta H \left(2 \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right), \quad
T^{yy} = 2 \bar\eta H \left(2 \frac{\partial v}{\partial y} + \frac{\partial u}{\partial x}\right), \quad
T^{xy} = \bar\eta H \left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\right).
```

``\bar\eta H = H \int_0^1 \eta \, d\zeta`` is the depth-integrated Glen viscosity (see
[Rheology and sliding](rheology_and_sliding.md)) and ``\beta = \tau_b / |u_b|`` the basal drag coefficient, scaled
by the grounded fraction at the velocity point.

## Depth-integrated viscosity approximation (DIVA)

`stress_balance = DepthIntegrated(shear = DIVAShear())` adds vertical shear driven by the basal stress
[Goldberg2011, Lipscomb2019](@citep). The effective strain rate in the viscosity includes
``\dot\varepsilon_{xz} = \tau_b^x (1 - \zeta) / 2\eta``, and the depth-averaged velocity feels the reduced drag

```math
\beta_\text{eff} = \frac{\beta}{1 + \beta F_2}, \qquad F_2 = H \int_0^1 \frac{(1 - \zeta)^2}{\eta} \, d\zeta .
```

After each solve the basal velocity ``u_b = \bar u / (1 + \beta F_2)`` and the profile
``u(\zeta) = u_b (1 + \beta F_1(\zeta))``, with ``F_1(\zeta) = H \int_0^\zeta (1 - \zeta')/\eta \, d\zeta'``, are
reconstructed. SSA is the special case ``F_2 = 0``. For uniform ``\eta``, ``\bar\eta H = \eta H`` and ``F_2 = H/(3\eta)``.

## Boundary conditions

- **Side walls.** Velocities normal to `Bounded` walls vanish. The tangential condition is
  `side_walls = FreeSlipWalls()` (no shear stress at wall corners) or `NoSlipWalls()` (zero tangential velocity at
  the wall, imposed with a mirror ghost) in `DepthIntegrated(; side_walls)`.
- **Calving fronts.** At a face with ice on one side only, the driving stress is replaced by the net front force
  ``\Delta \, g (\rho_i H^2 - \rho_w D^2) / 2`` of the ice cell, with ``D`` its depth below sea level, so the front
  feels the ocean pressure.
- **Ice-free cells** add no stiffness. Velocity points with ice-free cells on both sides are removed from the solve
  like walls, so the operator stays symmetric as the ice extent changes.
