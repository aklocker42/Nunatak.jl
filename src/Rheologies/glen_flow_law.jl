"""
$(TYPEDEF)

Glen's flow law `η = ½ A^(-1/n) ε̇ₑ^((1-n)/n)` with the regularised effective strain rate
`ε̇ₑ² = ... + ε̇₀²`. The rate factor `A` is a `Field` so that gradients with respect to it exist.

$(TYPEDFIELDS)
"""
struct GlenFlowLaw{A, FT}
    "rate factor `A` (Pa⁻ⁿ s⁻¹), a `Field` at `(Center, Center, Nothing)`"
    rate_factor :: A
    "flow-law exponent `n`"
    exponent :: FT
    "strain-rate regularisation `ε̇₀` (s⁻¹)"
    strain_rate_regularization :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `GlenFlowLaw` on `grid`. `rate_factor` is required and may be a number, an array,
a function `(x, y)`, or a `Field`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> rheology = GlenFlowLaw(grid; rate_factor = 1e-24);

julia> rheology.exponent
3.0
```
"""
function GlenFlowLaw(grid; rate_factor, exponent = 3, strain_rate_regularization = 1e-13)
    FT = eltype(grid)
    A = parameter_field(grid, rate_factor)
    return GlenFlowLaw(A, convert(FT, exponent), convert(FT, strain_rate_regularization))
end

Base.summary(r::GlenFlowLaw) = string("GlenFlowLaw(exponent = ", r.exponent, ")")

"""
$(TYPEDSIGNATURES)

Compute the Glen viscosity `ησ = ½ A^(-1/n) (ε̇ₑ² + ε̇₀²)^((1-n)/2n)` on the σ grid from the
depth-averaged velocities `u`, `v`, whose halos must be filled. The membrane strain rate is
uniform in the column, so `ησ` varies with depth only through `A`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_viscosity!

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat));

julia> σgrid = SigmaGrid(grid; Nσ = 3);

julia> ησ = CenterField(σgrid);

julia> u, v = XFaceField(grid), YFaceField(grid);

julia> compute_viscosity!(ησ, GlenFlowLaw(grid; rate_factor = 1, exponent = 1), u, v)

julia> ησ[1, 1, 1]
0.5
```
"""
function compute_viscosity!(ησ, rheology::GlenFlowLaw, u, v)
    grid = u.grid
    σgrid = ησ.grid
    A = rheology.rate_factor
    n = rheology.exponent
    ε̇₀ = rheology.strain_rate_regularization
    launch!(architecture(grid), σgrid, :xyz, _compute_viscosity!, ησ, u, v, A, n, ε̇₀, grid)
    return nothing
end

@kernel function _compute_viscosity!(ησ, u, v, A, n, ε̇₀, grid)
    i, j, k = @index(Global, NTuple)
    ε̇ₑ² = effective_strain_rate²(i, j, 1, grid, u, v) + ε̇₀^2
    @inbounds ησ[i, j, k] = A[i, j, 1]^(-1 / n) * ε̇ₑ²^((1 - n) / 2n) / 2
end

"""
$(TYPEDSIGNATURES)

Compute the Glen viscosity `ησ` with the DIVA vertical shear added to the membrane strain rate,
`ε̇ₑ² = ... + ε̇xz² + ε̇yz²` with `ε̇xz = τᵇˣ (1 - ζ) / 2η`. The basal stress `τᵇ = βᵇ uᵇ` comes from the
friction-law drag `βᵇᵘ`, `βᵇᵛ` and the basal velocities `uᵇ`, `vᵇ`, and `η` is the previous `ησ`,
so the shear lags one Picard iteration. Where `ησ` is still zero the shear is left out.
All velocity and drag halos must be filled.
"""
function compute_viscosity!(ησ, rheology::GlenFlowLaw, u, v, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ)
    grid = u.grid
    σgrid = ησ.grid
    A = rheology.rate_factor
    n = rheology.exponent
    ε̇₀ = rheology.strain_rate_regularization
    launch!(architecture(grid), σgrid, :xyz, _compute_shear_viscosity!, ησ, u, v, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ, A, n, ε̇₀, grid, σgrid)
    return nothing
end

@inline basal_stress(i, j, k, grid, β, u) = @inbounds β[i, j, k] * u[i, j, k]

@kernel function _compute_shear_viscosity!(ησ, u, v, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ, A, n, ε̇₀, grid, σgrid)
    i, j, k = @index(Global, NTuple)
    ζ = znode(i, j, k, σgrid, Center(), Center(), Center())
    η⁻ = @inbounds ησ[i, j, k]
    η⁻ = ifelse(iszero(η⁻), typemax(η⁻), η⁻)
    ε̇xz = ℑxᶜᵃᵃ(i, j, 1, grid, basal_stress, βᵇᵘ, uᵇ) * (1 - ζ) / 2η⁻
    ε̇yz = ℑyᵃᶜᵃ(i, j, 1, grid, basal_stress, βᵇᵛ, vᵇ) * (1 - ζ) / 2η⁻
    ε̇ₑ² = effective_strain_rate²(i, j, 1, grid, u, v) + ε̇xz^2 + ε̇yz^2 + ε̇₀^2
    @inbounds ησ[i, j, k] = A[i, j, 1]^(-1 / n) * ε̇ₑ²^((1 - n) / 2n) / 2
end
