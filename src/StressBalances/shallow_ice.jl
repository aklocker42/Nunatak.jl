"""
$(TYPEDEF)

Shallow-ice approximation (SIA): the velocity follows locally from the surface slope with no stress-balance solve,

`u(ζ) = uᵇ - 2 (ρᵢ g)ⁿ Hⁿ⁺¹ |∇s|ⁿ⁻¹ ∂s/∂x ∫₀^ζ A (1 - ζ′)ⁿ dζ′`,

evaluated at the velocity points with the face-averaged thickness and rate factor, the slope across the face
averaged from the four neighbouring differences. With `WeertmanFriction` the sliding velocity `uᵇ` balances
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

julia> ShallowIceSolver(schedule = IterationInterval(4)).schedule
IterationInterval(4, 0)
```
"""
ShallowIceSolver(; schedule = IterationInterval(1)) = ShallowIceSolver(schedule)

Base.summary(solver::ShallowIceSolver) = string("ShallowIceSolver(schedule = ", summary(solver.schedule), ")")

default_velocity_solver(grid, ::Nothing, stress_balance) = nothing
default_velocity_solver(grid, rheology, stress_balance) = StressBalanceSolver(grid)
default_velocity_solver(grid, ::Nothing, ::ShallowIce) = nothing
default_velocity_solver(grid, rheology, ::ShallowIce) = ShallowIceSolver()

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
##### Column integrals of the flow law
#####

# Exact integral of (1 - ζ)ⁿ from the bed to the centre of layer k, split into full layers below and the lower half of k.
@inline function glen_profile_weight(k′, k, σgrid, n)
    ζ⁻ = znode(1, 1, k′, σgrid, Center(), Center(), Face())
    ζ⁺ = ifelse(k′ == k, znode(1, 1, k, σgrid, Center(), Center(), Center()),
                          znode(1, 1, k′+1, σgrid, Center(), Center(), Face()))
    return ((1 - ζ⁻)^(n + 1) - (1 - ζ⁺)^(n + 1)) / (n + 1)
end

# Exact integral of (1 - ζ)ⁿ⁺¹ over layer k: the layer's share of the depth average of ∫₀^ζ (1 - ζ′)ⁿ dζ′.
@inline function glen_mean_weight(k, σgrid, n)
    ζ⁻ = znode(1, 1, k,   σgrid, Center(), Center(), Face())
    ζ⁺ = znode(1, 1, k+1, σgrid, Center(), Center(), Face())
    return ((1 - ζ⁻)^(n + 2) - (1 - ζ⁺)^(n + 2)) / (n + 2)
end

@inline x_face_rate_factor(i, j, k, A, structure) = (rate_factor(i-1, j, k, A, structure) + rate_factor(i, j, k, A, structure)) / 2
@inline y_face_rate_factor(i, j, k, A, structure) = (rate_factor(i, j-1, k, A, structure) + rate_factor(i, j, k, A, structure)) / 2

@inline function profile_integral(face_rate_factor, i, j, k, σgrid, A, structure, n)
    I = zero(n)
    for k′ in 1:k
        I += face_rate_factor(i, j, k′, A, structure) * glen_profile_weight(k′, k, σgrid, n)
    end
    return I
end

@inline function mean_integral(face_rate_factor, i, j, σgrid, A, structure, n)
    I = zero(n)
    for k in 1:size(σgrid, 3)
        I += face_rate_factor(i, j, k, A, structure) * glen_mean_weight(k, σgrid, n)
    end
    return I
end

#####
##### Face geometry, deformation and sliding
#####

# Thickness, slope along and squared slope at a velocity point.
@inline function x_face_slope(i, j, grid, H, s)
    Hᶠ = ℑxᶠᵃᵃ(i, j, 1, grid, H)
    ∂s = ∂xᶠᶜᶜ(i, j, 1, grid, s)
    ∂s⊥ = ℑxyᶠᶜᵃ(i, j, 1, grid, ∂yᶜᶠᶜ, s)
    return Hᶠ, ∂s, ∂s^2 + ∂s⊥^2
end

@inline function y_face_slope(i, j, grid, H, s)
    Hᶠ = ℑyᵃᶠᵃ(i, j, 1, grid, H)
    ∂s = ∂yᶜᶠᶜ(i, j, 1, grid, s)
    ∂s⊥ = ℑxyᶜᶠᵃ(i, j, 1, grid, ∂xᶠᶜᶜ, s)
    return Hᶠ, ∂s, ∂s^2 + ∂s⊥^2
end

# Deformation velocity per unit slope and per unit rate-factor integral, 2 (ρᵢ g)ⁿ Hⁿ⁺¹ |∇s|ⁿ⁻¹.
@inline deformation_coefficient(Hᶠ, S², n, ρᵢg) = 2 * ρᵢg^n * Hᶠ^(n + 1) * safe_power(S², (n - 1) / 2)

# Sliding velocity per unit slope: |uᵇ| = (ρᵢ g H |∇s| / C)^(1/m), directed down the slope; no sliding where C = 0.
@inline sliding_coefficient(Cᶠ, Hᶠ, S², ρᵢg, ::Nothing) = zero(Hᶠ)

@inline function sliding_coefficient(Cᶠ, Hᶠ, S², ρᵢg, friction::WeertmanFriction)
    p = 1 / friction.exponent
    C⁺ = ifelse(Cᶠ > 0, Cᶠ, one(Cᶠ))
    uᵇ = safe_power(ρᵢg * Hᶠ / C⁺, p) * safe_power(S², (p - 1) / 2)
    return ifelse(Cᶠ > 0, uᵇ, zero(uᵇ))
end

@inline x_friction_coefficient(i, j, grid, ::Nothing) = zero(grid)
@inline x_friction_coefficient(i, j, grid, friction) = ℑxᶠᵃᵃ(i, j, 1, grid, friction.coefficient)
@inline y_friction_coefficient(i, j, grid, ::Nothing) = zero(grid)
@inline y_friction_coefficient(i, j, grid, friction) = ℑyᵃᶠᵃ(i, j, 1, grid, friction.coefficient)

@inline function x_sliding_velocity(i, j, grid, Hᶠ, ∂s, S², φᵘ, friction, ρᵢg)
    uᵇ = - @inbounds(φᵘ[i, j, 1]) * sliding_coefficient(x_friction_coefficient(i, j, grid, friction), Hᶠ, S², ρᵢg, friction) * ∂s
    return ifelse(peripheral_node(i, j, 1, grid, Face(), Center(), Center()), zero(uᵇ), uᵇ)
end

@inline function y_sliding_velocity(i, j, grid, Hᶠ, ∂s, S², φᵛ, friction, ρᵢg)
    vᵇ = - @inbounds(φᵛ[i, j, 1]) * sliding_coefficient(y_friction_coefficient(i, j, grid, friction), Hᶠ, S², ρᵢg, friction) * ∂s
    return ifelse(peripheral_node(i, j, 1, grid, Center(), Face(), Center()), zero(vᵇ), vᵇ)
end

@inline function x_deformation_velocity(i, j, grid, Hᶠ, ∂s, S², φᵘ, n, ρᵢg, I)
    u = - @inbounds(φᵘ[i, j, 1]) * deformation_coefficient(Hᶠ, S², n, ρᵢg) * I * ∂s
    return ifelse(peripheral_node(i, j, 1, grid, Face(), Center(), Center()), zero(u), u)
end

@inline function y_deformation_velocity(i, j, grid, Hᶠ, ∂s, S², φᵛ, n, ρᵢg, I)
    v = - @inbounds(φᵛ[i, j, 1]) * deformation_coefficient(Hᶠ, S², n, ρᵢg) * I * ∂s
    return ifelse(peripheral_node(i, j, 1, grid, Center(), Face(), Center()), zero(v), v)
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
    launch!(arch, grid, size(u), _compute_x_shallow_ice_velocity!, u, uᵇ, grid, σgrid, H, s, φᵘ, A, structure, n, friction, ρᵢg)
    launch!(arch, grid, size(v), _compute_y_shallow_ice_velocity!, v, vᵇ, grid, σgrid, H, s, φᵛ, A, structure, n, friction, ρᵢg)
    launch!(arch, σgrid, size(uσ), _compute_x_shallow_ice_profile!, uσ, grid, σgrid, H, s, φᵘ, A, structure, n, friction, ρᵢg)
    launch!(arch, σgrid, size(vσ), _compute_y_shallow_ice_profile!, vσ, grid, σgrid, H, s, φᵛ, A, structure, n, friction, ρᵢg)
    return nothing
end

@kernel function _compute_x_shallow_ice_velocity!(u, uᵇ, grid, σgrid, H, s, φᵘ, A, structure, n, friction, ρᵢg)
    i, j = @index(Global, NTuple)
    Hᶠ, ∂s, S² = x_face_slope(i, j, grid, H, s)
    u₀ = x_sliding_velocity(i, j, grid, Hᶠ, ∂s, S², φᵘ, friction, ρᵢg)
    I = mean_integral(x_face_rate_factor, i, j, σgrid, A, structure, n)
    @inbounds uᵇ[i, j, 1] = u₀
    @inbounds u[i, j, 1] = u₀ + x_deformation_velocity(i, j, grid, Hᶠ, ∂s, S², φᵘ, n, ρᵢg, I)
end

@kernel function _compute_y_shallow_ice_velocity!(v, vᵇ, grid, σgrid, H, s, φᵛ, A, structure, n, friction, ρᵢg)
    i, j = @index(Global, NTuple)
    Hᶠ, ∂s, S² = y_face_slope(i, j, grid, H, s)
    v₀ = y_sliding_velocity(i, j, grid, Hᶠ, ∂s, S², φᵛ, friction, ρᵢg)
    I = mean_integral(y_face_rate_factor, i, j, σgrid, A, structure, n)
    @inbounds vᵇ[i, j, 1] = v₀
    @inbounds v[i, j, 1] = v₀ + y_deformation_velocity(i, j, grid, Hᶠ, ∂s, S², φᵛ, n, ρᵢg, I)
end

@kernel function _compute_x_shallow_ice_profile!(uσ, grid, σgrid, H, s, φᵘ, A, structure, n, friction, ρᵢg)
    i, j, k = @index(Global, NTuple)
    Hᶠ, ∂s, S² = x_face_slope(i, j, grid, H, s)
    I = profile_integral(x_face_rate_factor, i, j, k, σgrid, A, structure, n)
    @inbounds uσ[i, j, k] = x_sliding_velocity(i, j, grid, Hᶠ, ∂s, S², φᵘ, friction, ρᵢg) +
                            x_deformation_velocity(i, j, grid, Hᶠ, ∂s, S², φᵘ, n, ρᵢg, I)
end

@kernel function _compute_y_shallow_ice_profile!(vσ, grid, σgrid, H, s, φᵛ, A, structure, n, friction, ρᵢg)
    i, j, k = @index(Global, NTuple)
    Hᶠ, ∂s, S² = y_face_slope(i, j, grid, H, s)
    I = profile_integral(y_face_rate_factor, i, j, k, σgrid, A, structure, n)
    @inbounds vσ[i, j, k] = y_sliding_velocity(i, j, grid, Hᶠ, ∂s, S², φᵛ, friction, ρᵢg) +
                            y_deformation_velocity(i, j, grid, Hᶠ, ∂s, S², φᵛ, n, ρᵢg, I)
end

#####
##### Explicit diffusion limit on the time step
#####

# Thickness diffusivity over the cell area, D / Δ², with D = |ū| H / |∂s| the diffusivity of the face flux.
@inline function x_diffusion_rate(i, j, k, grid, u, H, s)
    Hᶠ, ∂s, _ = x_face_slope(i, j, grid, H, s)
    D = @inbounds abs(u[i, j, 1]) * Hᶠ / ifelse(∂s == 0, one(∂s), abs(∂s))
    return ifelse(∂s == 0, zero(D), D) / Δxᶠᶜᶜ(i, j, 1, grid)^2
end

@inline function y_diffusion_rate(i, j, k, grid, v, H, s)
    Hᶠ, ∂s, _ = y_face_slope(i, j, grid, H, s)
    D = @inbounds abs(v[i, j, 1]) * Hᶠ / ifelse(∂s == 0, one(∂s), abs(∂s))
    return ifelse(∂s == 0, zero(D), D) / Δyᶜᶠᶜ(i, j, 1, grid)^2
end

#####
##### Strain heating
#####

# Squared slope at a cell centre: mean of the squared face slopes, wall faces excluded.
@inline function cell_slope²(i, j, grid, s)
    w⁻ = !peripheral_node(i,   j, 1, grid, Face(), Center(), Center())
    w⁺ = !peripheral_node(i+1, j, 1, grid, Face(), Center(), Center())
    Sˣ = (w⁻ * ∂xᶠᶜᶜ(i, j, 1, grid, s)^2 + w⁺ * ∂xᶠᶜᶜ(i+1, j, 1, grid, s)^2) / max(w⁻ + w⁺, 1)
    w⁻ = !peripheral_node(i, j,   1, grid, Center(), Face(), Center())
    w⁺ = !peripheral_node(i, j+1, 1, grid, Center(), Face(), Center())
    Sʸ = (w⁻ * ∂yᶜᶠᶜ(i, j, 1, grid, s)^2 + w⁺ * ∂yᶜᶠᶜ(i, j+1, 1, grid, s)^2) / max(w⁻ + w⁺, 1)
    return Sˣ + Sʸ
end

# Layer mean of 2 A τₑⁿ⁺¹ with τₑ = ρᵢ g H (1 - ζ) |∇s|; the factor (1 - ζ)ⁿ⁺¹ is averaged exactly over the layer.
@kernel function _compute_shallow_ice_heating!(Φ, grid, σgrid, H, s, φ, A, structure, n, ρᵢg)
    i, j, k = @index(Global, NTuple)
    S² = cell_slope²(i, j, grid, s)
    τ = @inbounds ρᵢg * H[i, j, 1]
    weight = glen_mean_weight(k, σgrid, n) / layer_thickness(k, σgrid)
    @inbounds Φ[i, j, k] = φ[i, j, 1] * 2 * rate_factor(i, j, k, A, structure) * τ^(n + 1) * safe_power(S², (n + 1) / 2) * weight
end
