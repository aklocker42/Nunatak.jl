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
  neighbours, up to `passes` times per step. `passes = 0` holds the front fixed.
- The transport conserves volume. Surface mass balance and melt act on full cells during the step, and then on
  the ice of partial cells, which empty when the melt removes all of it.

## Calving

Nunatak has no calving law. A fixed calving front can be imposed in two ways. One is a `PartialCellFront` with
`passes = 0`. The other is a large melt rate in the ocean beyond the front, which removes any ice that flows there;
the [MISMIP+ benchmark](../literated/mismip_plus.md) does this.
