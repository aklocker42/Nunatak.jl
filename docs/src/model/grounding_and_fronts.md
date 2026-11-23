# Grounding lines and calving fronts

## Sub-grid grounded fraction

[`SubgridGroundedFraction`](@ref) locates the grounding line inside a cell from the flotation function

```math
f = H + \frac{\rho_w}{\rho_i} b ,
```

which is positive where the ice is grounded. ``f`` is interpolated linearly between cell centres, so the grounded
fraction ``\phi \in [0, 1]`` moves continuously as the grounding line crosses a cell. At a velocity point,
``\phi^u`` or ``\phi^v`` is the grounded part of the segment joining the two neighbouring centres; at a centre,
``\phi`` is the grounded part of the cell.

- Basal friction is scaled by ``\phi^u``, ``\phi^v``.
- Sub-shelf melt is scaled by ``1 - \phi``.
- The surface is ``s = \max(b + H, (1 - \rho_i/\rho_w) H)``.

With `SubgridGroundedFraction(driving_stress = true)` the driving stress on a velocity face whose two centres lie on
opposite sides of the grounding line is split there too. The grounding line sits at ``\theta = f_1 / (f_1 - f_2)`` along
the segment, where the ice is at flotation with thickness ``H^g = H_1 + \theta (H_2 - H_1)`` and surface
``s^g = (1 - \rho_i/\rho_w) H^g``. Each part contributes its own mean thickness times its own surface drop,

```math
\tau = -\Delta \, \rho_i g \, \frac{(H_1 + H^g)(s^g - s_1) + (H^g + H_2)(s_2 - s^g)}{2} ,
```

with ``\Delta`` the face width. This captures the kink of the surface at the grounding line instead of averaging
across it. The option is off by default.

Without a grounding treatment (`grounding = nothing`) all ice rests on the bed and ``s = b + H``.

## Ice masks

The ice mask ``\mu`` scales the depth-integrated viscosity, so cells with ``\mu = 0`` are ice free and drop out of the
stress balance.

**[`SmoothIceMask`](@ref)** (the default) rises from 0 at `minimum_thickness` to 1 at
`minimum_thickness + transition_thickness` as a cubic smoothstep, so it is continuously differentiable in ``H``. The
ice margin moves wherever the mass balance takes it.

**[`PartialCellFront`](@ref)** tracks a calving front with partially filled cells, after [Albrecht2011](@citet)
and as in MITgcm STREAMICE:

- ``\mu`` is a persistent 0/1 flag of full cells, and only full cells enter the stress balance.
- A cell that is not full but holds ice is a partial front cell. Ice of reference thickness ``H^r`` covers the
  fraction ``H / H^r`` of it. It receives thickness from full neighbours flowing into it and passes none on.
- At the end of each step ``H^r`` becomes the mean thickness of the full cells flowing in.
- A partial cell whose mean thickness reaches ``H^r`` becomes full. It passes the excess volume to its empty
  neighbours, up to `passes` times per step. `passes = 0` stops the advance.
- The transport conserves volume. Surface mass balance and melt act on full cells during the step, and then on
  the ice of partial cells, which empty when the melt removes all of it.

## Calving

A `PartialCellFront` takes a calving law through its `calving` keyword. Calving acts on marine front cells, those
that hold ice and have an empty neighbour with its bed below sea level. After the advance, a front cell of area
``A`` with ocean faces of total length ``\ell`` loses the fraction ``c \, \Delta t \, \ell / A`` of its area, at
the calving rate ``c``. It keeps its ice thickness, so a full cell that calves becomes partial. A partial cell
has no stress balance of its own and takes ``c`` from its full neighbours. The removed thickness accumulates in
`calved_thickness`, which closes the volume budget. The retreat is capped at one cell per step.

- **[`ThicknessThresholdCalving`](@ref)** removes, in one step, a front cell whose ice is thinner than `thickness`.
- **[`EigenCalving`](@ref)** [Levermann2012](@citep) retreats at ``c = K \max(\dot\varepsilon_1, 0) \max(\dot\varepsilon_2, 0)``,
  with ``\dot\varepsilon_{1,2}`` the principal horizontal strain rates.
- **[`VonMisesCalving`](@ref)** [Morlighem2016](@citep) retreats at ``c = |\mathbf{u}| \, \tilde\sigma / \sigma_\mathrm{max}``,
  with the tensile von Mises stress ``\tilde\sigma = \sqrt{3} B \tilde\varepsilon^{1/n}``,
  ``\tilde\varepsilon^2 = (\max(\dot\varepsilon_1, 0)^2 + \max(\dot\varepsilon_2, 0)^2) / 2``, ``B`` taken from the
  depth-averaged viscosity, and ``\sigma_\mathrm{max}`` blended between its grounded and floating values by the
  grounded fraction.

The square roots and the power ``1/n`` are regularized with the flow law's `strain_rate_regularization`, so the
rates stay differentiable where the ice does not deform.

A fixed calving front can also be imposed without a calving law: either a `PartialCellFront` with `passes = 0`, or
a large melt rate in the ocean beyond the front, which removes any ice that flows there; the
[MISMIP+ benchmark](../literated/mismip_plus.md) does this.
