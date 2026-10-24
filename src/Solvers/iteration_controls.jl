"""
$(TYPEDEF)

Iterate until the residual norm falls below `max(reltol ‖r₀‖, abstol)` or `maxiter` is reached.
The trip count depends on the data, so reverse-mode AD goes through a custom adjoint instead.

$(TYPEDFIELDS)
"""
struct Convergence{FT}
    "relative tolerance"
    reltol :: FT
    "absolute tolerance"
    abstol :: FT
    "maximum number of iterations"
    maxiter :: Int
end

"""
$(TYPEDSIGNATURES)

```jldoctest
julia> using Nunatak

julia> Convergence(; reltol = 1e-10, maxiter = 50)
Convergence{Float64}(1.0e-10, 0.0, 50)
```
"""
function Convergence(FT = Float64; reltol = sqrt(eps(FT)), abstol = 0, maxiter = 1000)
    return Convergence(convert(FT, reltol), convert(FT, abstol), maxiter)
end

"""
$(TYPEDEF)

Iterate exactly `iterations` times. A fixed trip count traces to straight-line code, which
Reactant reverse mode differentiates cheaply (pattern of Breeze.jl `Solvers.FixedIterations`).

$(TYPEDFIELDS)
"""
struct FixedIterations
    "number of iterations"
    iterations :: Int
end
