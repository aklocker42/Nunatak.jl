# Validation

The scripts in `benchmarks/` compare Nunatak with analytic solutions and published model results, in cases too slow
for the test suite. Run them from that directory with its environment, for example
`julia --project=. -e 'include("halfar.jl"); halfar()'`. The [examples](literated/ice_cap.md) are tutorials and make no
such comparisons.

| Script | Case | Reference |
|---|---|---|
| `halfar.jl` | Isothermal similarity dome, 25 kyr | [Halfar1983](@citet); [Bueler2005](@citet), Test B |
| `eismint2_a.jl` | Thermomechanical ice cap grown from zero ice, 200 kyr | [Payne2000](@citet), experiment A |
| `halfpipe_streamice.jl` | Floating shelf in a channel with an advancing calving front, 2 years | MITgcm STREAMICE `verification/halfpipe_streamice` |
| `halfpipe_streamice_adjoint.jl` | Gradient of a velocity and thickness cost with respect to basal melt | MITgcm STREAMICE `verification/halfpipe_streamice` (`input_ad`) |
| `mismip_plus.jl` | Marine ice stream on a retrograde bed: spin-up, Ice1r, Ice1ra, Ice2r, Ice2ra | [AsayDavis2016](@citet); STREAMICE in [Cornford2020](@citet) |
| `schoof_flowline.jl` | Steady grounding line of a marine flowline, approached from upstream and downstream | [Schoof2007](@citet) |
| `calvingmip.jl` | Circular and Thule domains, held and prescribed front motion, Exp1 to Exp4 | [CalvingMIP](https://github.com/JRowanJordan/CalvingMIP/wiki) |
| `kleiner2015_a.jl` | Slab under a surface temperature cycle, basal melt and refreezing | [Kleiner2015](@citet), experiment A |
| `kleiner2015_b.jl` | Steady polythermal slab, cold-temperate transition surface | [Kleiner2015](@citet), experiment B |

## Halfpipe against STREAMICE

A mostly floating shelf in a half-pipe channel, fed through the west edge, with no-slip side walls and a calving
front that advances through partially filled cells. Nunatak against STREAMICE run with converged Picard iterations:

| t (a) | Volume (km³), Nunatak / STREAMICE | Area (km²), Nunatak / STREAMICE | Centerline front (km), Nunatak / STREAMICE |
|---|---|---|---|
| 0.0 | 1209.73 / 1209.73 | 2660.0 / 2660.0 | 70.00 / 70.00 |
| 0.4 | 1232.53 / 1232.53 | 2671.8 / 2673.0 | 70.38 / 70.45 |
| 1.0 | 1266.73 / 1266.73 | 2694.9 / 2693.7 | 71.14 / 71.18 |
| 1.4 | 1289.53 / 1289.53 | 2710.6 / 2708.6 | 71.65 / 71.70 |
| 2.0 | 1323.73 / 1323.21 | 2734.4 / 2734.6 | 72.43 / 72.59 |

On the same geometry, the maximum speeds of the two models agree to 0.4%. STREAMICE's speed alternates between high
and low values from one step to the next, so per-step speeds are not compared.

## Grounding line on a flowline against Schoof (2007)

A marine flowline on a prograde bed, ``b = -100 - 6 \times 10^{-4} x`` m, with Weertman friction (``m = 1/3``,
``C = 10^4`` Pa m⁻¹ᐟ³ a¹ᐟ³), ``A = 2 \times 10^{-17}`` Pa⁻³ a⁻¹, accumulation 0.3 m a⁻¹ and no buttressing. The
boundary-layer theory of [Schoof2007](@citet) puts the steady grounding line at 408.0 km. Nunatak runs to steady
state from thin ice (advance) and from thick ice grounded past that position (retreat); with a continuum model both
would end at the same place, and the gap between them is the resolution error.

| Grid | Driving stress | Advance (km) | Retreat (km) |
|---|---|---|---|
| uniform 8 km | centred | 363.8 | 495.4 |
| uniform 8 km | split at the grounding line | 363.8 | 487.1 |
| 4 km between 280 and 560 km, 8 km outside | centred | 384.5 | 454.0 |
| 4 km between 280 and 560 km, 8 km outside | split at the grounding line | 384.5 | 446.1 |

Splitting the driving stress at the grounding line (`SubgridGroundedFraction(driving_stress = true)`) moves the
retreated position about 8 km closer at both resolutions and leaves the advance unchanged. Refining only the band the
grounding line crosses costs about half as much as refining the whole flowline.

## CalvingMIP

The circular and Thule domains of [CalvingMIP](https://github.com/JRowanJordan/CalvingMIP/wiki), on the
experiment grid of 5 km cells over ``\pm 800`` km. The bed is ``b = 900 - 2900\, r^2 / R^2`` m with ``R = 800`` km,
and Thule adds ``1100 \cos(3 \pi r / l)`` m with ``l = R - R \cos(2\theta) / 2``. The ice starts 100 m thick inside
750 km, with Glen ``A = 9.31 \times 10^{-26}`` Pa⁻³ s⁻¹, Weertman friction with ``m = 1/3``, accumulation
0.3 m a⁻¹ and densities 917 and 1028 kg m⁻³.

| Experiment | Domain | Calving | Length |
|---|---|---|---|
| Exp1 | circular | [`MaximumExtentCalving`](@ref) at 750 km | 10 000 a to steady state |
| Exp2 | circular | [`FrontVelocityCalving`](@ref), ``w = -300 \sin(2 \pi t / 1000\,\text{a})`` m a⁻¹ | 1000 a from Exp1 |
| Exp3 | Thule | [`MaximumExtentCalving`](@ref) at 750 km | 10 000 a to steady state |
| Exp4 | Thule | [`FrontVelocityCalving`](@ref), ``w = -750 \sin(2 \pi t / 1000\,\text{a})`` m a⁻¹ for 500 a, then [`MaximumExtentCalving`](@ref) at 750 km | 1000 a from Exp3 |

Grounded ice does not calve under [`FrontVelocityCalving`](@ref), as the protocol asks. In Exp4 the front advances
freely after 500 a, and the maximum extent stops it at its starting position.

The CalvingMIP outputs follow from the partially filled front cells. A cell with ice mask ``\mu = 0`` and thickness
``H`` below its front thickness ``H^r`` covers the fraction ``H / H^r`` of its area with ice ``H^r`` thick. A cell
counts as ice when it is at least half covered, and as grounded when its grounded fraction is at least one half. This
gives the mask (1 grounded, 2 floating, 3 ocean) and `lithk`. The areas weight each cell by its coverage. Along each
profile, the front is where the bilinearly interpolated coverage falls through one half.

| CalvingMIP variable | Nunatak |
|---|---|
| `lithk` | ``H`` in full cells, ``H^r`` in covered partial cells |
| `xvelmean`, `yvelmean` | depth-averaged velocity, from cell faces to centres |
| `iareagr`, `iareafl` | covered area with grounded fraction above and below one half |
| `lim`, `limnsw` | ``\rho_i \sum H\, \Delta x \Delta y``, and the grounded part above flotation |
| `tendlicalvf` | ``\rho_i`` times the volume calved between outputs, per year |
| `tendligroundf` | ice flux through the faces between grounded and floating cells |
| `xcf`, `ycf` on each profile | front position along the profile, from the coverage |

The front displacement is the change in distance from the centre of the front on each profile, in the direction the
prescribed front velocity acts. The Caprona profiles end about 742 km from the centre, inside the initial front, so
their displacement is measured from 750 km once the front crosses them. Exp2, mean and range over the eight profiles:

| t (a) | Front displacement (km) | Prescribed (km) | Floating area (10³ km²) | Mass (10¹⁸ kg) |
|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 845 | 1.037 |
| 250 | −47.7 (−48.5 to −47.0) | −47.7 | 628 | 0.973 |
| 500 | −93.8 (−94.8 to −92.9) | −95.5 | 424 | 0.910 |
| 750 | −47.0 (−47.9 to −46.0) | −47.7 | 620 | 0.967 |
| 1000 | −0.6 (−2.1 to 0.8) | 0.0 | 825 | 1.029 |

The grounded area stays at 921 × 10³ km² and the mass above flotation changes by less than 0.1 %.

In Exp4 the Halbrane fronts retreat 233 km by 481 a against the prescribed 239 km at 500 a. The Caprona fronts
retreat over floating ice but lag the prescribed motion by 15 to 20 %, reaching 189 km at 500 a. After 500 a the front advances freely and is back at 750 km on the Halbrane profiles at 860 a, held there
by the maximum extent. The grounded area falls from 937 to 892 × 10³ km² at 500 a and recovers to 918 × 10³ km².
