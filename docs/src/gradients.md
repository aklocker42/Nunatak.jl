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

The diagram shows which parts of a gradient Enzyme differentiates directly and where the custom rules take over.

```@raw html
<svg class="nunatak-diagram" viewBox="0 0 720 498" role="img" aria-label="Forward and reverse passes of a gradient through the model" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="nd-head-adj" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="nd-head" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<rect class="nd-key" x="10" y="10" width="700" height="40" rx="6"/>
<text class="nd-mono" x="360.0" y="26.5" text-anchor="middle">autodiff(Reverse, cost, Active, Duplicated(model, dmodel), …)</text>
<text class="nd-note" x="360.0" y="42.5" text-anchor="middle">or checkpointed_gradient!(dmodel, cost, model, Δt, Nt) for long runs</text>
<rect class="nd-tab" x="180" y="62" width="260" height="26" rx="6"/>
<text class="nd-note" x="310.0" y="79.5" text-anchor="middle">forward, top to bottom: run and tape</text>
<rect class="nd-tab" x="450" y="62" width="260" height="26" rx="6"/>
<text class="nd-note" x="580.0" y="79.5" text-anchor="middle">reverse, bottom to top: adjoint</text>
<rect class="nd-branch" x="10" y="100" width="160" height="50" rx="6"/>
<text class="nd-mono" x="90.0" y="113.5" text-anchor="middle">launch! kernels</text>
<text class="nd-note" x="90.0" y="129.5" text-anchor="middle">tendency, geometry, drag,</text>
<text class="nd-note" x="90.0" y="145.5" text-anchor="middle">viscosity, enthalpy</text>
<rect class="nd-box" x="180" y="100" width="260" height="50" rx="6"/>
<text class="nd-note" x="310.0" y="121.5" text-anchor="middle">every kernel runs as in a forward step</text>
<text class="nd-note" x="310.0" y="137.5" text-anchor="middle">Enzyme tapes the values it needs</text>
<rect class="nd-box" x="450" y="100" width="260" height="50" rx="6"/>
<text class="nd-note" x="580.0" y="121.5" text-anchor="middle">Enzyme runs the adjoint of each kernel</text>
<text class="nd-note" x="580.0" y="137.5" text-anchor="middle">no hand-written rules</text>
<rect class="nd-branch" x="10" y="162" width="160" height="82" rx="6"/>
<text class="nd-mono" x="90.0" y="199.5" text-anchor="middle">picard_solve!</text>
<text class="nd-note" x="90.0" y="215.5" text-anchor="middle">Convergence</text>
<rect class="nd-box" x="180" y="162" width="260" height="82" rx="6"/>
<text class="nd-note" x="310.0" y="191.5" text-anchor="middle">iterate to convergence</text>
<text class="nd-note" x="310.0" y="207.5" text-anchor="middle">tape the converged x and a snapshot</text>
<text class="nd-note" x="310.0" y="223.5" text-anchor="middle">of the coefficients only</text>
<rect class="nd-box" x="450" y="162" width="260" height="82" rx="6"/>
<text class="nd-note" x="580.0" y="183.5" text-anchor="middle">fixed-point adjoint, iterated to</text>
<text class="nd-note" x="580.0" y="199.5" text-anchor="middle">adjoint_iteration_control:</text>
<text class="nd-note" x="580.0" y="215.5" text-anchor="middle">picard_step_adjoint!, then Aᵀλ = x̄ + ȳ</text>
<text class="nd-note" x="580.0" y="231.5" text-anchor="middle">b̄ += λ; parameters from one last step</text>
<rect class="nd-branch" x="10" y="256" width="160" height="66" rx="6"/>
<text class="nd-mono" x="90.0" y="285.5" text-anchor="middle">picard_solve!</text>
<text class="nd-note" x="90.0" y="301.5" text-anchor="middle">FixedIterations(N)</text>
<rect class="nd-box" x="180" y="256" width="260" height="66" rx="6"/>
<text class="nd-note" x="310.0" y="277.5" text-anchor="middle">no custom rule</text>
<text class="nd-note" x="310.0" y="293.5" text-anchor="middle">the N iterations are taped as</text>
<text class="nd-note" x="310.0" y="309.5" text-anchor="middle">straight-line code</text>
<rect class="nd-box" x="450" y="256" width="260" height="66" rx="6"/>
<text class="nd-note" x="580.0" y="285.5" text-anchor="middle">Enzyme reverses the loop</text>
<text class="nd-note" x="580.0" y="301.5" text-anchor="middle">each solve! uses the rule below</text>
<rect class="nd-branch" x="10" y="334" width="160" height="82" rx="6"/>
<text class="nd-mono" x="90.0" y="371.5" text-anchor="middle">solve!</text>
<text class="nd-note" x="90.0" y="387.5" text-anchor="middle">conjugate gradients</text>
<rect class="nd-box" x="180" y="334" width="260" height="82" rx="6"/>
<text class="nd-note" x="310.0" y="371.5" text-anchor="middle">solve A(p) x = b</text>
<text class="nd-note" x="310.0" y="387.5" text-anchor="middle">tape x and snapshot(p)</text>
<rect class="nd-box" x="450" y="334" width="260" height="82" rx="6"/>
<text class="nd-note" x="580.0" y="355.5" text-anchor="middle">adjoint_solve! Aᵀλ = x̄, same solver</text>
<text class="nd-note" x="580.0" y="371.5" text-anchor="middle">b̄ += λ</text>
<text class="nd-note" x="580.0" y="387.5" text-anchor="middle">p̄ −= (∂(A x)/∂p)ᵀ λ by Enzyme</text>
<text class="nd-note" x="580.0" y="403.5" text-anchor="middle">on the operator</text>
<rect class="nd-key" x="10" y="432" width="700" height="58" rx="6"/>
<text class="nd-mono" x="360.0" y="449.5" text-anchor="middle">checkpointed_gradient!</text>
<text class="nd-note" x="360.0" y="465.5" text-anchor="middle">forward: store prognostic_state every checkpoint_interval steps</text>
<text class="nd-note" x="360.0" y="481.5" text-anchor="middle">reverse: restore each checkpoint and differentiate its segment, last first, into dmodel</text>
</svg>
```

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
