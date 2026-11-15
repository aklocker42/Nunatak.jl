# Gradients and adjoints

Nunatak is written so that [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) can differentiate whole
simulations in reverse mode. Every parameter that can vary in space (the bed, the friction coefficient, the rate
factor, the mass balance, the geothermal flux) is a `Field` inside the model. A single reverse pass therefore gives
the gradient of a scalar cost with respect to all of them and to the initial thickness. The rules for the solvers
live in the `NunatakEnzymeExt` package extension, which loads with `using Enzyme`.

## Differentiating a function of the model

Write the cost as a function of the model, make a zero shadow, and call `autodiff`:

```julia
using Oceananigans, Nunatak, Enzyme

function cost(model, uᵒ)
    update_state!(model)
    u = model.velocities.u
    J = zero(eltype(u))
    for j in 1:size(uᵒ, 2), i in 1:size(uᵒ, 1)
        @inbounds J += (u[i, j, 1] - uᵒ[i, j])^2
    end
    return J
end

dmodel = Enzyme.make_zero(model)
_, J = autodiff(set_runtime_activity(ReverseWithPrimal), cost, Active, Duplicated(model, dmodel), Const(uᵒ))

∂J_∂C = dmodel.friction.coefficient     # gradient with respect to the friction coefficient
∂J_∂b = dmodel.bed                      # ... the bed
```

The [friction inversion example](literated/friction_inversion.md) uses this with L-BFGS to recover a basal friction
field from noisy surface velocities.

## The velocity solve

The adjoint of the stress balance is not obtained by taping the solver's iterations.

- **Linear solves.** For ``x = A(p)^{-1} b``, the custom rule solves ``A^T \lambda = \bar{x}`` with the same
  conjugate-gradient solver, then accumulates ``\bar{b} \mathrel{+}= \lambda`` and
  ``\bar{p} \mathrel{-}= (\partial (A x) / \partial p)^T \lambda``. The operator is self-adjoint, so the adjoint
  solve reuses the forward operator.
- **Picard with `Convergence`.** Reverse mode differentiates the converged solution rather than the iterates. The
  adjoint is the fixed point of the adjoint of one Picard step [Christianson1994](@citep), iterated to
  `adjoint_iteration_control`, so the tape holds only the converged state.
- **Picard with `FixedIterations(N)`.** The trip count is fixed, so the iterations are differentiated as
  straight-line code, which also suits Reactant.

## Through time: checkpointed gradients

For a cost evaluated after many time steps, [`checkpointed_gradient!`](@ref) avoids taping the whole run:

```julia
dmodel = Enzyme.make_zero(model)
J = checkpointed_gradient!(dmodel, cost, model, Δt, Nt; checkpoint_interval = ceil(Int, sqrt(Nt)))
```

- The forward run stores the prognostic state every `checkpoint_interval` steps.
- The reverse pass restores each checkpoint and differentiates its segment, so only one segment is taped at a time.
- The gradient with respect to the initial state and the parameter fields accumulates in `dmodel`.
- `model` is left at the final state.

The test suite checks these gradients against finite differences, through converged Picard time steps and through
energy steps. The [STREAMICE halfpipe adjoint benchmark](literated/halfpipe_streamice_adjoint.md) compares them with
the TAF-generated adjoint of MITgcm STREAMICE.
