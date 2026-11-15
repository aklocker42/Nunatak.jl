# Thermodynamics

[`EnthalpyThermodynamics`](@ref) follows [Aschwanden2012](@citet): one prognostic enthalpy ``E`` per layer of the
σ grid covers both cold ice and temperate ice. Defaults follow PISM and Table A1 of [Kleiner2015](@citet).

## Enthalpy and temperature

With the pressure-melting temperature ``T_{pm} = T_0 - \beta p`` and the enthalpy at the melting point
``E_{pm} = c_i (T_{pm} - T_r)``,

```math
T = \begin{cases} T_r + E / c_i & E < E_{pm} \text{ (cold)} \\ T_{pm} & E \ge E_{pm} \text{ (temperate)} \end{cases},
\qquad
\omega = \frac{\max(E - E_{pm}, 0)}{L} .
```

The temperature and water fraction set the rate factor when the rheology uses an [`ArrheniusRateFactor`](@ref).

## Energy equation

The enthalpy is advanced once per ice time step, after the velocity solve, with one implicit tridiagonal solve per
column. In finite-volume form on layer ``k``,

```math
\rho \frac{H^{+} E^{+} - H^{n} E^{n}}{\Delta t} \Delta\zeta + \rho \Delta\zeta \, \nabla \cdot (H^n \mathbf{u} E^n)
+ Q_{k+1} - Q_k = H^{+} \Delta\zeta \, \Phi .
```

- Horizontal transport is explicit and first-order upwind, so an energy step needs ``u \Delta t / \Delta x \le 1``.
- ``Q`` holds vertical advection and conduction ``-K \partial E / \partial z``. The volume flux makes a uniform
  ``E`` an exact solution. Harmonic face averages of ``K`` keep the flux consistent across the cold–temperate
  transition.
- The strain heating ``\Phi = 4 \eta \dot\varepsilon_e^2`` uses the physical strain rate, without the flow-law
  regularization. Under DIVA it includes the vertical shear, whose column integral is ``\tau_b^2 F_2``. Under the
  shallow-ice approximation it is ``2 A \tau_e^{n+1}`` with ``\tau_e = \rho_i g H (1 - \zeta) |\nabla s|``.

The energy step can be thinned with `schedule = IterationInterval(N)`. Its time step is then the time elapsed
since the last energy step.

## Base

- Basal melt water is stored in a local layer ``H_w`` without transport. It refreezes when the column draws heat
  from the base.
- Water above `drainage_water_fraction` drains from temperate layers into ``H_w``.
- A temperate base under a temperate layer is insulating [Kleiner2015; Eq. 6](@citep), so all basal heat melts ice
  there.
- With `grounded_melt_coupling = true`, the change of ``H_w`` is taken from the ice thickness, so ice and water mass
  is conserved.

Set the initial state with `set!(model; E)` or `set!(model; T)`, as a function of ``(x, y, \zeta)``, a number or a
`Field`. Otherwise the enthalpy starts at the surface value. The
[polythermal slab benchmarks](../literated/kleiner2015_b.md) test the formulation against [Kleiner2015](@citet).
