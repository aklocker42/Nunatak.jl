```@raw html
---
layout: home

hero:
  name: "Nunatak.jl"
  tagline: "Differentiable ice-sheet and ice-shelf modelling in Julia, on CPUs and GPUs."
  image:
    src: /logo.svg
    alt: Nunatak
  actions:
    - theme: brand
      text: Get Started
      link: /model/overview
    - theme: alt
      text: View on Github
      link: https://github.com/aklocker42/Nunatak.jl
    - theme: brand
      text: Validation
      link: /validation

features:
  - title: Stress balances
    details: Shallow ice, shallow shelf and depth-integrated viscosity (DIVA) on a staggered C-grid.
    link: /model/stress_balances
  - title: Grounding lines and fronts
    details: Sub-grid grounded fractions, calving-front stresses and partially filled front cells.
    link: /model/grounding_and_fronts
  - title: Thermodynamics
    details: Enthalpy with cold and temperate ice, basal water and strain heating on a σ grid.
    link: /model/thermodynamics
  - title: Gradients
    details: Enzyme reverse mode through whole simulations, with a fixed-point adjoint of the velocity solve.
    link: /gradients
---
```

Nunatak is a differentiable ice-sheet and ice-shelf model built on
[Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl). It runs on CPUs and GPUs, and Enzyme computes
gradients through whole simulations for inversions and sensitivity studies.

## Features

- **Stress balances**: shallow ice (SIA), the shallow-shelf approximation, and depth-integrated viscosity
  (DIVA, `DIVAShear()`), on a staggered C-grid with no-slip or free-slip side walls.
- **Velocity solver**: Picard iterations around a preconditioned conjugate-gradient solve, with Jacobi, diagonal
  or geometric-multigrid preconditioners.
- **Rheology and sliding**: Glen's flow law with a constant or Arrhenius rate factor, and Weertman or regularized
  Coulomb friction, with an effective pressure from the ocean connection or from till water.
- **Grounding lines and calving fronts**: subgrid grounded fractions, calving-front stresses, and a front that
  advances through partially filled cells.
- **Thermodynamics**: enthalpy with cold and temperate ice, basal water and strain heating, on a terrain-following
  σ-grid.
- **Adjoints**: Enzyme reverse-mode rules for the velocity solve, including a fixed-point adjoint of the Picard
  iteration, plus checkpointed gradients through time stepping.

## Installation

Nunatak is not registered yet. Add it from its repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/aklocker42/Nunatak.jl")
```

## Example

A [Halfar1983](@citet) dome evolving under the shallow-ice approximation:

```julia
using Oceananigans, Oceananigans.Units, Nunatak

year = years = 31556926seconds
grid = RectilinearGrid(CPU(); size = (97, 97), halo = (4, 4),
                       x = (-1212.5kilometers, 1212.5kilometers), y = (-1212.5kilometers, 1212.5kilometers),
                       topology = (Bounded, Bounded, Flat))

model = IceSheetModel(grid; Nσ = 10, stress_balance = ShallowIce(),
                      rheology = GlenFlowLaw(grid; rate_factor = 1e-16 / year, exponent = 3),
                      friction = WeertmanFriction(grid; coefficient = 1e18, exponent = 1))

set!(model, H = (x, y) -> 3600 * max(0, 1 - (hypot(x, y) / 750kilometers)^(4 / 3))^(3 / 7))

simulation = Simulation(model; Δt = 1year, stop_time = 100years)
run!(simulation)
```

`examples/` has tutorials: an ice cap grown with shallow ice, a marine ice sheet advancing into the ocean, and a
basal friction inversion with Enzyme gradients.

See [Validation](validation.md) for benchmarks against analytic solutions and other models.

## Citing

See `CITATION.cff` in the repository.

## License

MIT, see `LICENSE` in the repository.
