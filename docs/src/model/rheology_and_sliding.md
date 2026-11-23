# Rheology and sliding

## Glen's flow law

[`GlenFlowLaw`](@ref) gives the effective viscosity

```math
\eta = \tfrac{1}{2} A^{-1/n} \left(\dot\varepsilon_e^2 + \dot\varepsilon_0^2\right)^{(1-n)/2n},
\qquad
\dot\varepsilon_e^2 = \dot\varepsilon_{xx}^2 + \dot\varepsilon_{yy}^2 + \dot\varepsilon_{xx}\dot\varepsilon_{yy} + \dot\varepsilon_{xy}^2 ,
```

plus the vertical shear terms under DIVA. ``\dot\varepsilon_0`` (`strain_rate_regularization`) keeps the viscosity
finite where the ice does not deform. The membrane strain rate is uniform in the column, so the viscosity varies with
depth only through the rate factor ``A``.

`rate_factor` may be

- a number, an array, a function `(x, y)` or a `Field` for a depth-independent ``A``,
- a `CenterField` on the σ grid for a prescribed depth-varying ``A``,
- an [`ArrheniusRateFactor`](@ref), for ``A`` that follows the enthalpy (requires `EnthalpyThermodynamics`).

The Arrhenius law is that of [PatersonBudd1982](@citet) with the water softening of [LliboutryDuval1985](@citet), as
written by [GreveBlatter2009](@citet):

```math
A = A_0 \exp\left(-\frac{Q}{R T'}\right) \left(1 + c \min(\omega, \omega_\text{max})\right),
```

at the pressure-adjusted temperature ``T'`` and water fraction ``\omega``. The logarithm of ``A`` is blended between
the cold and warm branches with a smoothstep, which removes the jump of about 1% at the transition temperature and
keeps the derivative continuous. Defaults are PISM's. The law holds for ``n = 3``.

## Basal sliding

The friction law sets the drag coefficient ``\beta = \tau_b / |u_b|`` at the velocity points from the basal
speed, which is regularized as ``\sqrt{|u_b|^2 + u_0^2}`` (`velocity_regularization`). The drag is multiplied by the
grounded fraction at the velocity point, so floating ice slides freely.

**[`WeertmanFriction`](@ref)**: a power law with exponent ``m`` (default ``1/3``),

```math
\tau_b = C |u_b|^{m-1} u_b .
```

**[`RegularizedCoulombFriction`](@ref)**: the law of [Schoof2005](@citet) used in MISMIP+ [AsayDavis2016](@citep),

```math
\tau_b = \frac{C |u_b|^m \, \alpha^2 N}{\left(C^{1/m} |u_b| + (\alpha^2 N)^{1/m}\right)^m} \frac{u_b}{|u_b|},
```

which tends to Weertman ``C |u_b|^m`` for slow sliding and to Coulomb ``\alpha^2 N`` for fast sliding. Because the
effective pressure ``N`` vanishes at the grounding line, friction goes smoothly to zero there.

By default ``N = \max(\rho_i g H + \rho_w g \min(b, 0), 0)``, the effective pressure of a bed connected to the
ocean. `EnthalpyThermodynamics(grid; effective_pressure = TillEffectivePressure(grid))` sets it instead from the
basal water ``H_w`` of the thermodynamics, after the till model of [BuelerVanPelt2015](@citet):

```math
N = \min\left(N_0 \left(\frac{\delta P_o}{N_0}\right)^s 10^{(e_0 / C_c)(1 - s)}, \; N^o\right), \qquad
s = \min\left(\frac{H_w}{W_\mathrm{max}}, 1\right),
```

with the overburden ``P_o = \rho_i g H`` and ``N^o`` the ocean-connected value, which keeps ``N = 0`` at flotation.
A dry till carries ``N_0 10^{e_0 / C_c}``, a saturated one ``\delta P_o``. The defaults are
``N_0 = 1000`` Pa, ``\delta = 0.02``, ``e_0 = 0.69``, ``C_c = 0.12`` and ``W_\mathrm{max} = 2`` m. The till
water drains at ``C_d = 1`` mm per year and is capped at ``W_\mathrm{max}``; the drained and the excess water
leave the model, as in the non-conserving till model of PISM.

The friction coefficient ``C`` is a `Field`, so it can vary in space and be inverted for; see the
[friction inversion example](../literated/friction_inversion.md).
