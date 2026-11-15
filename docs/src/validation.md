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
