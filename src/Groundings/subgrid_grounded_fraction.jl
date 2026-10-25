"""
$(TYPEDEF)

Sub-grid grounded fraction from the flotation function `f = H + ρw b / ρᵢ` (sea level at `z = 0`),
interpolated linearly between cell centres, so `φ ∈ [0, 1]` moves continuously as the grounding
line crosses a cell. At a velocity point `φᵘ`, `φᵛ` is the grounded part of the segment joining the
two neighbouring centres; at a centre `φ` is the grounded part of the cell.
Friction is scaled by the face fractions `φᵘ`, `φᵛ` and melt by `1 - φ`. The densities come from
`IceSheetConstants`.
"""
struct SubgridGroundedFraction end

Base.summary(::SubgridGroundedFraction) = "SubgridGroundedFraction"

@inline flotation(i, j, k, grid, H, b, ρw_ρᵢ) = @inbounds H[i, j, k] + ρw_ρᵢ * b[i, j, k]

# Fraction of a segment with f > 0 when f varies linearly from f₁ to f₂.
@inline function segment_grounded_fraction(f₁, f₂)
    f⁺ = max(f₁, zero(f₁)) + max(f₂, zero(f₂))
    Σf = abs(f₁) + abs(f₂)
    return ifelse(Σf == 0, one(f₁), f⁺ / nonzero(Σf))
end

# Fraction of a triangle with f > 0 when f is linear between vertex values f₁, f₂, f₃.
@inline function triangle_grounded_fraction(f₁, f₂, f₃)
    a = min(f₁, f₂, f₃)
    c = max(f₁, f₂, f₃)
    b = f₁ + f₂ + f₃ - a - c
    one_grounded  = c^2 / nonzero((c - a) * (c - b))
    one_floating = one(a) - a^2 / nonzero((b - a) * (c - a))
    φ = ifelse(b > 0, one_floating, one_grounded)
    return ifelse(a >= 0, one(a), ifelse(c <= 0, zero(a), φ))
end

@inline nonzero(x) = ifelse(x == 0, one(x), x)

# Eight triangles (centre, edge midpoint, corner) tile the cell; f is linear on each.
@inline function grounded_fractionᶜᶜᶜ(i, j, k, grid, H, b, ρw_ρᵢ)
    f(i′, j′) = flotation(i′, j′, k, grid, H, b, ρw_ρᵢ)
    fᶜ = f(i, j)
    fᵉ, fʷ, fⁿ, fˢ = (fᶜ + f(i + 1, j)) / 2, (fᶜ + f(i - 1, j)) / 2, (fᶜ + f(i, j + 1)) / 2, (fᶜ + f(i, j - 1)) / 2
    fⁿᵉ = (fᶜ + f(i + 1, j) + f(i, j + 1) + f(i + 1, j + 1)) / 4
    fⁿʷ = (fᶜ + f(i - 1, j) + f(i, j + 1) + f(i - 1, j + 1)) / 4
    fˢᵉ = (fᶜ + f(i + 1, j) + f(i, j - 1) + f(i + 1, j - 1)) / 4
    fˢʷ = (fᶜ + f(i - 1, j) + f(i, j - 1) + f(i - 1, j - 1)) / 4
    φ = triangle_grounded_fraction(fᶜ, fᵉ, fⁿᵉ) + triangle_grounded_fraction(fᶜ, fⁿ, fⁿᵉ) +
        triangle_grounded_fraction(fᶜ, fⁿ, fⁿʷ) + triangle_grounded_fraction(fᶜ, fʷ, fⁿʷ) +
        triangle_grounded_fraction(fᶜ, fʷ, fˢʷ) + triangle_grounded_fraction(fᶜ, fˢ, fˢʷ) +
        triangle_grounded_fraction(fᶜ, fˢ, fˢᵉ) + triangle_grounded_fraction(fᶜ, fᵉ, fˢᵉ)
    return φ / 8
end

@inline grounded_fractionᶠᶜᶜ(i, j, k, grid, H, b, ρw_ρᵢ) =
    segment_grounded_fraction(flotation(i - 1, j, k, grid, H, b, ρw_ρᵢ), flotation(i, j, k, grid, H, b, ρw_ρᵢ))

@inline grounded_fractionᶜᶠᶜ(i, j, k, grid, H, b, ρw_ρᵢ) =
    segment_grounded_fraction(flotation(i, j - 1, k, grid, H, b, ρw_ρᵢ), flotation(i, j, k, grid, H, b, ρw_ρᵢ))

"""
$(TYPEDSIGNATURES)

Compute the grounded fraction `φ` at cell centres and `φᵘ`, `φᵛ` at the velocity points from the
thickness `H` and bed `b`. `H` and `b` need filled halos. Without a grounding treatment
(`nothing`) all ice is grounded.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_grounded_fractions!

julia> using Oceananigans.BoundaryConditions: fill_halo_regions!

julia> grid = RectilinearGrid(size = (4, 1), x = (0, 4), y = (0, 1), topology = (Bounded, Periodic, Flat));

julia> H, b = CenterField(grid), CenterField(grid);

julia> set!(H, 100); set!(b, (x, y) -> -100 * (x - 0.75)); fill_halo_regions!((H, b));

julia> φ, φᵘ, φᵛ = CenterField(grid), XFaceField(grid), YFaceField(grid);

julia> constants = IceSheetConstants(grid; ice_density = 1, ocean_density = 1);

julia> compute_grounded_fractions!(φ, φᵘ, φᵛ, SubgridGroundedFraction(), H, b, constants)

julia> collect(interior(φ, :, 1, 1))
4-element Vector{Float64}:
 1.0
 0.75
 0.0
 0.0
```
"""
function compute_grounded_fractions!(φ, φᵘ, φᵛ, ::SubgridGroundedFraction, H, b, constants)
    grid = H.grid
    arch = architecture(grid)
    ρw_ρᵢ = constants.ocean_density / constants.ice_density
    launch!(arch, grid, size(φ),  _compute_grounded_fraction!,   φ,  H, b, ρw_ρᵢ, grid)
    launch!(arch, grid, size(φᵘ), _compute_x_grounded_fraction!, φᵘ, H, b, ρw_ρᵢ, grid)
    launch!(arch, grid, size(φᵛ), _compute_y_grounded_fraction!, φᵛ, H, b, ρw_ρᵢ, grid)
    return nothing
end

function compute_grounded_fractions!(φ, φᵘ, φᵛ, ::Nothing, args...)
    fill!(φ, 1)
    fill!(φᵘ, 1)
    fill!(φᵛ, 1)
    return nothing
end

@kernel function _compute_grounded_fraction!(φ, H, b, ρw_ρᵢ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds φ[i, j, k] = grounded_fractionᶜᶜᶜ(i, j, k, grid, H, b, ρw_ρᵢ)
end

@kernel function _compute_x_grounded_fraction!(φᵘ, H, b, ρw_ρᵢ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds φᵘ[i, j, k] = grounded_fractionᶠᶜᶜ(i, j, k, grid, H, b, ρw_ρᵢ)
end

@kernel function _compute_y_grounded_fraction!(φᵛ, H, b, ρw_ρᵢ, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds φᵛ[i, j, k] = grounded_fractionᶜᶠᶜ(i, j, k, grid, H, b, ρw_ρᵢ)
end

"""
$(TYPEDSIGNATURES)

Compute the effective pressure `N = max(ρ_i g H + ρ_w g min(b, 0), 0)` at cell centres, for a bed
connected to the ocean. `N` is zero at and seaward of the grounding line, so the Coulomb branch of
`RegularizedCoulombFriction` takes friction smoothly to zero there.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_effective_pressure!

julia> grid = RectilinearGrid(size = (2, 1), x = (0, 2), y = (0, 1), topology = (Bounded, Periodic, Flat));

julia> H, b, N = CenterField(grid), CenterField(grid), CenterField(grid);

julia> set!(H, 10); set!(b, (x, y) -> x < 1 ? 5 : -5);

julia> compute_effective_pressure!(N, H, b, IceSheetConstants(grid; ice_density = 1, ocean_density = 1, gravitational_acceleration = 1))

julia> collect(interior(N, :, 1, 1))
2-element Vector{Float64}:
 10.0
  5.0
```
"""
function compute_effective_pressure!(N, H, b, constants)
    grid = H.grid
    ρᵢg = constants.ice_density * constants.gravitational_acceleration
    ρwg = constants.ocean_density * constants.gravitational_acceleration
    launch!(architecture(grid), grid, size(N), _compute_effective_pressure!, N, H, b, ρᵢg, ρwg, grid)
    return nothing
end

@kernel function _compute_effective_pressure!(N, H, b, ρᵢg, ρwg, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds N[i, j, k] = max(ρᵢg * H[i, j, k] + ρwg * min(b[i, j, k], zero(grid)), zero(grid))
end

"""
$(TYPEDSIGNATURES)

Compute the surface elevation `s = max(b + H, (1 - ρᵢ / ρw) H)`: on the bed where the ice is
grounded and in hydrostatic balance with the ocean where it floats. Without a grounding
treatment (`nothing`) the ice rests on the bed, `s = b + H`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> using Nunatak: compute_surface_elevation!

julia> grid = RectilinearGrid(size = (2, 1), x = (0, 2), y = (0, 1), topology = (Bounded, Periodic, Flat));

julia> H, b, s = CenterField(grid), CenterField(grid), CenterField(grid);

julia> set!(H, 100); set!(b, (x, y) -> x < 1 ? -20 : -500);

julia> compute_surface_elevation!(s, SubgridGroundedFraction(), H, b, IceSheetConstants(grid; ice_density = 1, ocean_density = 2))

julia> collect(interior(s, :, 1, 1))
2-element Vector{Float64}:
 80.0
 50.0
```
"""
function compute_surface_elevation!(s, ::SubgridGroundedFraction, H, b, constants)
    grid = H.grid
    ρᵢ_ρw = constants.ice_density / constants.ocean_density
    launch!(architecture(grid), grid, size(s), _compute_floating_surface_elevation!, s, H, b, ρᵢ_ρw, grid)
    return nothing
end

function compute_surface_elevation!(s, ::Nothing, H, b, constants)
    grid = H.grid
    launch!(architecture(grid), grid, size(s), _compute_grounded_surface_elevation!, s, H, b)
    return nothing
end

@kernel function _compute_floating_surface_elevation!(s, H, b, ρᵢ_ρw, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds s[i, j, k] = max(b[i, j, k] + H[i, j, k], (1 - ρᵢ_ρw) * H[i, j, k])
end

@kernel function _compute_grounded_surface_elevation!(s, H, b)
    i, j, k = @index(Global, NTuple)
    @inbounds s[i, j, k] = b[i, j, k] + H[i, j, k]
end
