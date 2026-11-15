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

The friction coefficient ``C`` is a `Field`, so it can vary in space and be inverted for; see the
[friction inversion example](../literated/friction_inversion.md).
