# Validation

The scripts in `benchmarks/` run longer cases than the test suite. Run them from that directory with the
examples environment, for example `julia --project=../examples -e 'include("halfar.jl"); halfar()'`.

| Script | Case | Reference |
|---|---|---|
| `halfar.jl` | Isothermal similarity dome, 25 kyr | Halfar (1983); Bueler et al. (2005), Test B |
| `eismint2_a.jl` | Thermomechanical ice cap grown from zero ice, 200 kyr | Payne et al. (2000), experiment A |
| `halfpipe_streamice.jl` | Floating shelf in a channel with an advancing calving front, 2 years | MITgcm STREAMICE `verification/halfpipe_streamice` |

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
