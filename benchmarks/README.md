# Benchmarks

Runs against analytic solutions and published model results, too slow for the test suite. The tutorials for
users are in `../examples`. Run from this directory with its environment, e.g.

```
julia --project=. -e 'include("halfar.jl"); halfar()'
julia --project=. -e 'include("eismint2_a.jl"); eismint2_a(Nσ = 40)'
julia --project=. -e 'include("halfpipe_streamice.jl"); halfpipe_streamice()'
julia --project=. -e 'include("halfpipe_streamice_adjoint.jl"); halfpipe_streamice_adjoint()'
julia --project=. -e 'include("mismip_plus.jl"); mismip_plus()'
julia --project=. -e 'include("schoof_flowline.jl"); schoof_flowline(start = :retreat, driving_stress = true)'
julia --project=. kleiner2015_a.jl
julia --project=. kleiner2015_b.jl
```

`plot_benchmarks.jl` turns the CSV output into the figures on the documentation's validation pages.

`halfar.jl` and `eismint2_a.jl` take a `stress_balance` keyword (default `ShallowIce()`); pass `DepthIntegrated(shear = DIVAShear())`
to run the same case with DIVA and a no-slip bed. Each run writes `<output>_timeseries.csv` and the final
fields as CSV files.

| Script | Case | Reference |
|---|---|---|
| `halfar.jl` | Isothermal similarity dome, 25 kyr from t₀ | Halfar (1983); Bueler et al. (2005, J. Glaciol. 51(173)) Test B |
| `eismint2_a.jl` | Thermomechanical ice cap grown from zero ice, 200 kyr | Payne et al. (2000, J. Glaciol. 46(153)), experiment A |
| `halfpipe_streamice.jl` | Floating shelf in a channel with an advancing calving front, 2 years | MITgcm STREAMICE `verification/halfpipe_streamice` |
| `halfpipe_streamice_adjoint.jl` | Gradient of a velocity and thickness cost with respect to basal melt, 3 steps | MITgcm STREAMICE `verification/halfpipe_streamice` (`input_ad`) |
| `mismip_plus.jl` | Marine ice stream on a retrograde bed: spin-up and melt experiments Ice1r, Ice1ra, Ice2r, Ice2ra | Asay-Davis et al. (2016, Geosci. Model Dev. 9); STREAMICE in Cornford et al. (2020, The Cryosphere 14) |
| `schoof_flowline.jl` | Steady grounding line of a marine flowline approached from upstream and downstream, uniform or stretched Δx | Schoof (2007, J. Geophys. Res. 112) |
| `kleiner2015_a.jl` | Slab under a surface temperature cycle, basal melt and refreezing | Kleiner et al. (2015, The Cryosphere 9), experiment A |
| `kleiner2015_b.jl` | Steady polythermal slab, cold-temperate transition surface | Kleiner et al. (2015, The Cryosphere 9), experiment B |

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

`mismip_plus_streamice_reference.csv` holds volume, volume above flotation, grounded area and centerline grounding-line
position for Ice0 and the four perturbation experiments from the MITgcm STREAMICE submission to Cornford et al. (2020),
whose rate factor and Schoof sliding law `mismip_plus.jl` uses.
