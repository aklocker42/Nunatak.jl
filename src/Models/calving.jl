"""
$(TYPEDEF)

Calving law for a [`PartialCellFront`](@ref) that removes, in one step, all the ice of a marine front cell whose
ice is thinner than `thickness`.

$(TYPEDFIELDS)
"""
struct ThicknessThresholdCalving{FT}
    "thickness below which a front cell calves (m)"
    thickness :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `ThicknessThresholdCalving` that calves front cells thinner than `thickness`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> ThicknessThresholdCalving(grid; thickness = 200)
ThicknessThresholdCalving(thickness = 200.0)
```
"""
ThicknessThresholdCalving(grid::AbstractGrid; thickness) = ThicknessThresholdCalving(convert(eltype(grid), thickness))

"""
$(TYPEDEF)

Eigencalving law (Levermann et al., 2012, The Cryosphere 6, 273–286): the front retreats at
`c = K max(ε̇₁, 0) max(ε̇₂, 0)`, with `ε̇₁`, `ε̇₂` the principal horizontal strain rates.

$(TYPEDFIELDS)
"""
struct EigenCalving{FT}
    "proportionality constant `K` (m s)"
    coefficient :: FT
end

"""
$(TYPEDSIGNATURES)

Build an `EigenCalving` law with proportionality constant `coefficient` (m s).

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> EigenCalving(grid; coefficient = 1e17)
EigenCalving(coefficient = 1.0e17)
```
"""
EigenCalving(grid::AbstractGrid; coefficient) = EigenCalving(convert(eltype(grid), coefficient))

"""
$(TYPEDEF)

Von Mises calving law (Morlighem et al., 2016, Geophys. Res. Lett. 43, 2659–2666): the front retreats at
`c = |u| σ̃ / σᵐᵃˣ`, with the tensile von Mises stress `σ̃ = √3 B ε̃^(1/n)`,
`ε̃² = (max(ε̇₁, 0)² + max(ε̇₂, 0)²) / 2`, and `σᵐᵃˣ` blended between its grounded and floating values by the
grounded fraction.

$(TYPEDFIELDS)
"""
struct VonMisesCalving{FT}
    "stress threshold `σᵐᵃˣ` of grounded ice (Pa)"
    stress_threshold :: FT
    "stress threshold `σᵐᵃˣ` of floating ice (Pa)"
    floating_stress_threshold :: FT
end

"""
$(TYPEDSIGNATURES)

Build a `VonMisesCalving` law with grounded threshold `stress_threshold` and floating threshold
`floating_stress_threshold` (Pa).

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> VonMisesCalving(grid; stress_threshold = 1e6, floating_stress_threshold = 3e5)
VonMisesCalving(stress_threshold = 1.0e6, floating_stress_threshold = 300000.0)
```
"""
function VonMisesCalving(grid::AbstractGrid; stress_threshold, floating_stress_threshold = stress_threshold)
    FT = eltype(grid)
    return VonMisesCalving(convert(FT, stress_threshold), convert(FT, floating_stress_threshold))
end

"""
$(TYPEDEF)

Calving law that sets the velocity of the front (CalvingMIP, Experiments 2 and 4): floating front cells retreat at
`c = max(|u| - w, 0)`, with `|u|` the ice speed and `w` the prescribed front velocity, positive for advance. The front
holds still for `w = 0`. Grounded ice does not calve.

$(TYPEDFIELDS)
"""
struct FrontVelocityCalving{W}
    "front velocity `w` (m s⁻¹), a number or a function of time"
    front_velocity :: W
end

"""
$(TYPEDSIGNATURES)

Build a `FrontVelocityCalving` with front velocity `front_velocity` (m s⁻¹), a number or a function `w(t)` of time.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> FrontVelocityCalving(grid; front_velocity = 1e-5)
FrontVelocityCalving(front_velocity = 1.0e-5)
```
"""
FrontVelocityCalving(grid::AbstractGrid; front_velocity = 0) = FrontVelocityCalving(front_velocity_parameter(grid, front_velocity))

front_velocity_parameter(grid, w::Number) = convert(eltype(grid), w)
front_velocity_parameter(grid, w) = w

"""
$(TYPEDEF)

Calving law that removes, in one step, all the ice of a marine front cell outside the region where `extent` is
positive (CalvingMIP, Experiments 1, 3 and 4): the front cannot advance beyond `extent`.

$(TYPEDFIELDS)
"""
struct MaximumExtentCalving{E}
    "region the ice may cover, positive inside, a `Field` at `(Center, Center, Nothing)`"
    extent :: E
end

"""
$(TYPEDSIGNATURES)

Build a `MaximumExtentCalving` that keeps the ice where `extent` (a number, function of `(x, y)` or `Field`) is
positive.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> MaximumExtentCalving(grid; extent = (x, y) -> x < 0.5)
MaximumExtentCalving()
```
"""
MaximumExtentCalving(grid::AbstractGrid; extent) = MaximumExtentCalving(parameter_field(grid, extent))

Adapt.adapt_structure(to, law::MaximumExtentCalving) = MaximumExtentCalving(adapt(to, law.extent))

Base.summary(law::ThicknessThresholdCalving) = string("ThicknessThresholdCalving(thickness = ", law.thickness, ")")
Base.summary(law::EigenCalving) = string("EigenCalving(coefficient = ", law.coefficient, ")")
Base.summary(law::VonMisesCalving) = string("VonMisesCalving(stress_threshold = ", law.stress_threshold,
                                            ", floating_stress_threshold = ", law.floating_stress_threshold, ")")
Base.summary(law::FrontVelocityCalving) = string("FrontVelocityCalving(front_velocity = ", law.front_velocity, ")")
Base.summary(::MaximumExtentCalving) = "MaximumExtentCalving()"

const CalvingLaw = Union{ThicknessThresholdCalving, EigenCalving, VonMisesCalving, FrontVelocityCalving, MaximumExtentCalving}

Base.show(io::IO, law::CalvingLaw) = print(io, summary(law))

#####
##### Calving rates of full cells
#####

# `ε̇₀` keeps the square root differentiable where the strain rate vanishes.
@inline function principal_strain_rates(i, j, k, grid, u, v, μ, walls, ε̇₀)
    ε̇xx = strain_rate_xx(i, j, k, grid, u, μ)
    ε̇yy = strain_rate_yy(i, j, k, grid, v, μ)
    ε̇xy = ℑxyᶜᶜᵃ(i, j, k, grid, strain_rate_xy, u, v, walls, μ)
    ε̄ = (ε̇xx + ε̇yy) / 2
    r = sqrt(((ε̇xx - ε̇yy) / 2)^2 + ε̇xy^2 + ε̇₀^2)
    return ε̄ + r, ε̄ - r
end

@inline function calving_rate(i, j, k, grid, law::EigenCalving, μ, u, v, η̄H, H, φ, walls, n, ε̇₀, t)
    ε̇₁, ε̇₂ = principal_strain_rates(i, j, k, grid, u, v, μ, walls, ε̇₀)
    return law.coefficient * max(ε̇₁, zero(grid)) * max(ε̇₂, zero(grid))
end

# `B` follows from the depth-averaged viscosity; `ε̃² (ε̃² + ε̇₀²)^((1 - 2n) / 2n)` is `ε̃^(1/n)`, smooth at `ε̃ = 0`.
@inline function calving_rate(i, j, k, grid, law::VonMisesCalving, μ, u, v, η̄H, H, φ, walls, n, ε̇₀, t)
    ε̇₁, ε̇₂ = principal_strain_rates(i, j, k, grid, u, v, μ, walls, ε̇₀)
    ε̃² = (max(ε̇₁, zero(grid))^2 + max(ε̇₂, zero(grid))^2) / 2
    ε̇ₑ² = effective_strain_rate²(i, j, k, grid, u, v, walls) + ε̇₀^2
    @inbounds begin
        Hᵢ = H[i, j, k]
        η̄ = η̄H[i, j, k] / ifelse(Hᵢ > 0, Hᵢ, one(grid))
        σᵐᵃˣ = φ[i, j, k] * law.stress_threshold + (1 - φ[i, j, k]) * law.floating_stress_threshold
    end
    B = 2η̄ * ε̇ₑ²^((n - 1) / 2n)
    σ̃ = √3 * B * ε̃² * (ε̃² + ε̇₀^2)^((1 - 2n) / 2n)
    ū = ℑxᶜᵃᵃ(i, j, k, grid, interior_u, u, μ)
    v̄ = ℑyᵃᶜᵃ(i, j, k, grid, interior_v, v, μ)
    speed = sqrt(ū^2 + v̄^2 + (ε̇₀ * Δxᶜᶜᶜ(i, j, k, grid))^2)
    return speed * σ̃ / σᵐᵃˣ
end

@inline front_velocity(w::Number, t) = w
@inline front_velocity(w, t) = w(t)

@inline function calving_rate(i, j, k, grid, law::FrontVelocityCalving, μ, u, v, η̄H, H, φ, walls, n, ε̇₀, t)
    ū = ℑxᶜᵃᵃ(i, j, k, grid, interior_u, u, μ)
    v̄ = ℑyᵃᶜᵃ(i, j, k, grid, interior_v, v, μ)
    speed = sqrt(ū^2 + v̄^2 + (ε̇₀ * Δxᶜᶜᶜ(i, j, k, grid))^2)
    c = max(speed - front_velocity(law.front_velocity, t), zero(grid))
    return @inbounds (1 - φ[i, j, k]) * c
end

@inline neighbor_calving_rate(i, j, k, grid, law, μ, args...) =
    ifelse(full_neighbor(i, j, k, grid, μ), calving_rate(i, j, k, grid, law, μ, args...), zero(grid))

@inline neighbor_count(i, j, k, grid, μ) = ifelse(full_neighbor(i, j, k, grid, μ), 1, 0)

# A partial cell has no stress balance of its own: it takes the mean rate of its full neighbors.
@inline function front_calving_rate(i, j, k, grid, law, μ, args...)
    Σ = neighbor_calving_rate(i-1, j, k, grid, law, μ, args...) + neighbor_calving_rate(i+1, j, k, grid, law, μ, args...) +
        neighbor_calving_rate(i, j-1, k, grid, law, μ, args...) + neighbor_calving_rate(i, j+1, k, grid, law, μ, args...)
    n = neighbor_count(i-1, j, k, grid, μ) + neighbor_count(i+1, j, k, grid, μ) +
        neighbor_count(i, j-1, k, grid, μ) + neighbor_count(i, j+1, k, grid, μ)
    return ifelse(full_cell(i, j, k, μ), calving_rate(i, j, k, grid, law, μ, args...), Σ / max(n, 1))
end

#####
##### Fraction of a front cell removed in one step
#####

@inline retreat_fraction(i, j, k, grid, law::ThicknessThresholdCalving, Δt, ℓ, Hᶠ, μ, args...) =
    ifelse(Hᶠ < law.thickness, one(grid), zero(grid))

@inline retreat_fraction(i, j, k, grid, law::MaximumExtentCalving, Δt, ℓ, Hᶠ, μ, args...) =
    @inbounds ifelse(law.extent[i, j, k] > 0, zero(grid), one(grid))

# The front retreats by `c Δt` along each of its ocean faces, of total length `ℓ`.
@inline function retreat_fraction(i, j, k, grid, law, Δt, ℓ, Hᶠ, μ, args...)
    c = front_calving_rate(i, j, k, grid, law, μ, args...)
    return min(one(grid), c * Δt * ℓ / Azᶜᶜᶜ(i, j, k, grid))
end

# Rate laws carry the retreat a front cell could not take to the ice behind it; removal laws do not.
@inline residual_retreat(law, r) = r
@inline residual_retreat(::Union{ThicknessThresholdCalving, MaximumExtentCalving}, r) = zero(r)
