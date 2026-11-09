"""
$(TYPEDEF)

Shallow-ice approximation (SIA): the velocity follows locally from the surface slope with no stress-balance solve,

`u(ζ) = uᵇ - 2 (ρᵢ g)ⁿ Hⁿ⁺¹ |∇s|ⁿ⁻¹ ∂s/∂x ∫₀^ζ A (1 - ζ′)ⁿ dζ′`,

evaluated at the velocity points with the face-averaged thickness and rate factor, the slope across the face
averaged from the four neighboring differences. With `WeertmanFriction` the sliding velocity `uᵇ` balances
the driving stress, `C |uᵇ|ᵐ = ρᵢ g H |∇s|`; without friction, and at faces where `C = 0`, the bed is frozen. Floating ice is held still
through the grounded fractions `φᵘ`, `φᵛ`. The default thickness advection is then `Centered(order = 2)`,
so the thickness flux is the diffusive flux `-D ∇s` of Mahaffy (1976).

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (8, 8), halo = (4, 4), x = (0, 1e5), y = (0, 1e5), topology = (Bounded, Bounded, Flat));

julia> model = IceSheetModel(grid; stress_balance = ShallowIce(), rheology = GlenFlowLaw(grid; rate_factor = 1e-24));

julia> summary(model.velocity_solver)
"ShallowIceSolver(schedule = IterationInterval(1))"
```
"""
struct ShallowIce end

Base.summary(::ShallowIce) = "ShallowIce"

"""
$(TYPEDEF)

Velocity update of the `ShallowIce` stress balance: one explicit evaluation, no iteration.

$(TYPEDFIELDS)
"""
struct ShallowIceSolver{S}
    "schedule of the velocity updates in `update_state!`; the velocities stay fixed in between"
    schedule :: S
end

"""
$(TYPEDSIGNATURES)

```jldoctest
julia> using Oceananigans, Nunatak

julia> solver = ShallowIceSolver(schedule = IterationInterval(4));

julia> summary(solver)
"ShallowIceSolver(schedule = IterationInterval(4))"
```
"""
ShallowIceSolver(; schedule = IterationInterval(1)) = ShallowIceSolver(schedule)

Base.summary(solver::ShallowIceSolver) = string("ShallowIceSolver(schedule = ", summary(solver.schedule), ")")

default_velocity_solver(grid, ::Nothing, stress_balance) = nothing
default_velocity_solver(grid, rheology, stress_balance) = StressBalanceSolver(grid)
default_velocity_solver(grid, ::GlenFlowLaw, ::ShallowIce) = ShallowIceSolver()

validate_stress_balance(stress_balance, rheology, friction) = nothing
validate_stress_balance(::ShallowIce, ::Union{Nothing, GlenFlowLaw}, ::Union{Nothing, WeertmanFriction}) = nothing
validate_stress_balance(::ShallowIce, rheology, friction) =
    throw(ArgumentError("ShallowIce needs a GlenFlowLaw and either no friction or WeertmanFriction"))

validate_velocity_solver(stress_balance, velocity_solver) = nothing
validate_velocity_solver(::ShallowIce, ::Nothing) = nothing
validate_velocity_solver(::ShallowIce, ::ShallowIceSolver) = nothing
validate_velocity_solver(::ShallowIce, velocity_solver) =
    throw(ArgumentError("ShallowIce needs velocity_solver = ShallowIceSolver() or nothing"))
validate_velocity_solver(stress_balance, ::ShallowIceSolver) =
    throw(ArgumentError("ShallowIceSolver needs stress_balance = ShallowIce()"))

# xᵉ with the zero-slope and zero-thickness limits taken by value, so that neither the result nor its derivative is NaN.
@inline function safe_power(x, e)
    x⁺ = ifelse(x > 0, x, one(x))
    return ifelse(x > 0, x⁺^e, ifelse(iszero(e), one(x), zero(x)))
end

#####
##### Face geometry, deformation and sliding
#####

@inline face_rate_factor(i, j, k, A, structure, ::Face, ::Center) = (rate_factor(i-1, j, k, A, structure) + rate_factor(i, j, k, A, structure)) / 2
@inline face_rate_factor(i, j, k, A, structure, ::Center, ::Face) = (rate_factor(i, j-1, k, A, structure) + rate_factor(i, j, k, A, structure)) / 2

@inline face_value(i, j, grid, ϕ, ::Face, ::Center) = ℑxᶠᵃᵃ(i, j, 1, grid, ϕ)
@inline face_value(i, j, grid, ϕ, ::Center, ::Face) = ℑyᵃᶠᵃ(i, j, 1, grid, ϕ)

@inline face_spacing(i, j, grid, ::Face, ::Center) = Δxᶠᶜᶜ(i, j, 1, grid)
@inline face_spacing(i, j, grid, ::Center, ::Face) = Δyᶜᶠᶜ(i, j, 1, grid)

# Surface slope normal to the face and the squared slope with the tangential component averaged onto the face.
@inline function face_slope(i, j, grid, s, ::Face, ::Center)
    ∂s = ∂xᶠᶜᶜ(i, j, 1, grid, s)
    return ∂s, ∂s^2 + ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, s)^2
end

@inline function face_slope(i, j, grid, s, ::Center, ::Face)
    ∂s = ∂yᶜᶠᶜ(i, j, 1, grid, s)
    return ∂s, ∂s^2 + ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, s)^2
end

# Deformation velocity per unit slope and per unit rate-factor integral, 2 (ρᵢ g)ⁿ Hⁿ⁺¹ |∇s|ⁿ⁻¹.
@inline deformation_coefficient(Hᶠ, S², n, ρᵢg) = 2 * ρᵢg^n * Hᶠ^(n + 1) * safe_power(S², (n - 1) / 2)

# Sliding velocity per unit slope: |uᵇ| = (ρᵢ g H |∇s| / C)^(1/m), directed down the slope; no sliding where C = 0.
@inline sliding_coefficient(i, j, grid, Hᶠ, S², ρᵢg, ::Nothing, ℓx, ℓy) = zero(Hᶠ)

@inline function sliding_coefficient(i, j, grid, Hᶠ, S², ρᵢg, friction::WeertmanFriction, ℓx, ℓy)
    Cᶠ = face_value(i, j, grid, friction.coefficient, ℓx, ℓy)
    p = 1 / friction.exponent
    C⁺ = ifelse(Cᶠ > 0, Cᶠ, one(Cᶠ))
    uᵇ = safe_power(ρᵢg * Hᶠ / C⁺, p) * safe_power(S², (p - 1) / 2)
    return ifelse(Cᶠ > 0, uᵇ, zero(uᵇ))
end

# Velocity down the slope at a floating-masked face, zero on walls.
@inline function face_velocity(i, j, grid, φ, coefficient, ∂s, ℓx, ℓy)
    u = - @inbounds(φ[i, j, 1]) * coefficient * ∂s
    return ifelse(peripheral_node(i, j, 1, grid, ℓx, ℓy, Center()), zero(u), u)
end

"""
$(TYPEDSIGNATURES)

Compute the shallow-ice velocities from the thickness `H` and surface `s`, whose halos must be filled:
the sliding velocities `uᵇ`, `vᵇ`, the layer velocities `uσ`, `vσ` and their exact depth averages `u`, `v`.
"""
function compute_shallow_ice_velocities!(velocities, H, s, φᵘ, φᵛ, rheology, friction, ρᵢg)
    (; u, v, uᵇ, vᵇ, uσ, vσ) = velocities
    grid = H.grid
    σgrid = uσ.grid
    arch = architecture(grid)
    A = rate_factor_field(rheology.rate_factor)
    structure = depth_structure(A)
    n = rheology.exponent
    launch!(arch, grid, size(u), _compute_shallow_ice_velocity!, u, uᵇ, grid, σgrid, H, s, φᵘ, A, structure, n, friction, ρᵢg, Face(), Center())
    launch!(arch, grid, size(v), _compute_shallow_ice_velocity!, v, vᵇ, grid, σgrid, H, s, φᵛ, A, structure, n, friction, ρᵢg, Center(), Face())
    launch!(arch, σgrid, size(uσ), _compute_shallow_ice_profile!, uσ, grid, σgrid, H, s, φᵘ, A, structure, n, friction, ρᵢg, Face(), Center())
    launch!(arch, σgrid, size(vσ), _compute_shallow_ice_profile!, vσ, grid, σgrid, H, s, φᵛ, A, structure, n, friction, ρᵢg, Center(), Face())
    return nothing
end

# The depth average of ∫₀^ζ A (1 - ζ′)ⁿ dζ′ is ∫₀¹ A (1 - ζ)ⁿ⁺¹ dζ.
@kernel function _compute_shallow_ice_velocity!(u, uᵇ, grid, σgrid, H, s, φ, A, structure, n, friction, ρᵢg, ℓx, ℓy)
    i, j = @index(Global, NTuple)
    Hᶠ = face_value(i, j, grid, H, ℓx, ℓy)
    ∂s, S² = face_slope(i, j, grid, s, ℓx, ℓy)
    I = column_integral(face_rate_factor, i, j, σgrid, n + 1, A, structure, ℓx, ℓy)
    u₀ = face_velocity(i, j, grid, φ, sliding_coefficient(i, j, grid, Hᶠ, S², ρᵢg, friction, ℓx, ℓy), ∂s, ℓx, ℓy)
    @inbounds uᵇ[i, j, 1] = u₀
    @inbounds u[i, j, 1] = u₀ + face_velocity(i, j, grid, φ, deformation_coefficient(Hᶠ, S², n, ρᵢg) * I, ∂s, ℓx, ℓy)
end

@kernel function _compute_shallow_ice_profile!(uσ, grid, σgrid, H, s, φ, A, structure, n, friction, ρᵢg, ℓx, ℓy)
    i, j, k = @index(Global, NTuple)
    Hᶠ = face_value(i, j, grid, H, ℓx, ℓy)
    ∂s, S² = face_slope(i, j, grid, s, ℓx, ℓy)
    I = profile_integral(face_rate_factor, i, j, k, σgrid, n, A, structure, ℓx, ℓy)
    coefficient = sliding_coefficient(i, j, grid, Hᶠ, S², ρᵢg, friction, ℓx, ℓy) + deformation_coefficient(Hᶠ, S², n, ρᵢg) * I
    @inbounds uσ[i, j, k] = face_velocity(i, j, grid, φ, coefficient, ∂s, ℓx, ℓy)
end

#####
##### Explicit diffusion limit on the time step
#####

# Thickness diffusivity over the cell area, D / Δ², with D = |ū| H / |∂s| the diffusivity of the face flux.
@inline function diffusion_rate(i, j, k, grid, u, H, s, ℓx, ℓy)
    Hᶠ = face_value(i, j, grid, H, ℓx, ℓy)
    ∂s, _ = face_slope(i, j, grid, s, ℓx, ℓy)
    D = @inbounds abs(u[i, j, 1]) * Hᶠ / ifelse(∂s == 0, one(∂s), abs(∂s))
    return ifelse(∂s == 0, zero(D), D) / face_spacing(i, j, grid, ℓx, ℓy)^2
end

#####
##### Strain heating
#####

# Squared slope at a cell center: mean of the squared face slopes, faces on walls or next to ice-free cells excluded.
@inline function cell_slope²(i, j, grid, H, s)
    w⁻ = @inbounds !peripheral_node(i,   j, 1, grid, Face(), Center(), Center()) & (H[i-1, j, 1] > 0)
    w⁺ = @inbounds !peripheral_node(i+1, j, 1, grid, Face(), Center(), Center()) & (H[i+1, j, 1] > 0)
    Sˣ = (w⁻ * ∂xᶠᶜᶜ(i, j, 1, grid, s)^2 + w⁺ * ∂xᶠᶜᶜ(i+1, j, 1, grid, s)^2) / max(w⁻ + w⁺, 1)
    w⁻ = @inbounds !peripheral_node(i, j,   1, grid, Center(), Face(), Center()) & (H[i, j-1, 1] > 0)
    w⁺ = @inbounds !peripheral_node(i, j+1, 1, grid, Center(), Face(), Center()) & (H[i, j+1, 1] > 0)
    Sʸ = (w⁻ * ∂yᶜᶠᶜ(i, j, 1, grid, s)^2 + w⁺ * ∂yᶜᶠᶜ(i, j+1, 1, grid, s)^2) / max(w⁻ + w⁺, 1)
    return Sˣ + Sʸ
end

# Layer mean of 2 A τₑⁿ⁺¹ with τₑ = ρᵢ g H (1 - ζ) |∇s|; the factor (1 - ζ)ⁿ⁺¹ is averaged exactly over the layer.
@kernel function _compute_shallow_ice_heating!(Φ, grid, σgrid, H, s, φ, A, structure, n, ρᵢg)
    i, j, k = @index(Global, NTuple)
    S² = cell_slope²(i, j, grid, H, s)
    τ = @inbounds ρᵢg * H[i, j, 1]
    weight = layer_weight(k, σgrid, n + 1) / layer_thickness(k, σgrid)
    @inbounds Φ[i, j, k] = φ[i, j, 1] * 2 * rate_factor(i, j, k, A, structure) * τ^(n + 1) * safe_power(S², (n + 1) / 2) * weight
end
