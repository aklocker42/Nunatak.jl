# Benchmarks

Long runs against published solutions, too slow for the test suite. Run from this directory with the
examples environment, e.g.

```
julia --project=../examples -e 'include("halfar.jl"); halfar()'
julia --project=../examples -e 'include("eismint2_a.jl"); eismint2_a(Nσ = 40)'
julia --project=../examples -e 'include("halfpipe_streamice.jl"); halfpipe_streamice()'
```

`halfar.jl` and `eismint2_a.jl` take a `stress_balance` keyword (default `ShallowIce()`); pass `DepthIntegrated(shear = DIVAShear())`
to run the same case with DIVA and a no-slip bed. Each run writes `<output>_timeseries.csv` and the final
fields as CSV files.

| Script | Case | Reference |
|---|---|---|
| `halfar.jl` | Isothermal similarity dome, 25 kyr from t₀ | Halfar (1983); Bueler et al. (2005, J. Glaciol. 51(173)) Test B |
| `eismint2_a.jl` | Thermomechanical ice cap grown from zero ice, 200 kyr | Payne et al. (2000, J. Glaciol. 46(153)), experiment A |
| `halfpipe_streamice.jl` | Floating shelf in a channel with an advancing calving front, 2 years | MITgcm STREAMICE `verification/halfpipe_streamice` |

EISMINT II experiment A after 200 kyr, intercomparison mean and range across ten models (Payne et al., 2000, Table 4):

| | Volume (10⁶ km³) | Area (10⁶ km²) | Melt fraction | Divide thickness (m) | Divide basal temperature (K) |
|---|---|---|---|---|---|
| Mean | 2.128 | 1.034 | 0.718 | 3688.342 | 255.605 |
| Range | 0.145 | 0.086 | 0.290 | 96.740 | 2.929 |

`halfpipe_streamice_reference.csv` holds STREAMICE's volume, area, centerline front position and maximum speed after
each 0.2-year step, from the MITgcm experiment rerun with converged Picard iterations (`streamice_max_nl_iter = 500`).
STREAMICE solves for the velocity at the start of each step, so each speed belongs to the geometry in the same row.
Its speed alternates between high and low values from one step to the next, so compare the geometry columns and treat
the speeds as a range.
