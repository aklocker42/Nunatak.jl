"""
$(TYPEDEF)

Calving-front advance with partially filled front cells (Albrecht et al., 2011, The Cryosphere 5, 35–44),
as in MITgcm STREAMICE. The ice mask `μ` is a persistent 0/1 flag of full cells, and only full cells enter
the stress balance. A cell that is not full but holds ice is a partial front cell: ice of thickness `Hʳ`
(`geometry.front_thickness`) covers the fraction `H / Hʳ` of it. It receives thickness from full neighbors
flowing into it and passes none on. At the end of each time step `Hʳ` becomes the mean thickness of the full
cells flowing in, or is kept when none does; a cell holding ice without `Hʳ` takes the mean thickness of its
full neighbors, or its own thickness when it has none. A partial cell whose mean thickness reaches `Hʳ` becomes full with
`H = Hʳ` and splits the excess volume equally among its empty neighbors; this repeats up to `passes` times.
Excess left after the last pass stays in the receiving cells, so the transport conserves volume. Full cells that
drain to `H = 0` become empty. Surface mass balance and basal melt act on full cells during the step and then,
as in STREAMICE, on the ice of the partial cells and of the cells that filled, which empty when the melt
removes all of it.

$(TYPEDFIELDS)
"""
struct PartialCellFront{F}
    "volume `O` an overfilled front cell passes to each empty neighbor (m³)"
    overflow :: F
    "maximum number of overflow passes per time step"
    passes :: Int
end

"""
$(TYPEDSIGNATURES)

Build a `PartialCellFront` with its overflow scratch field on `grid`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> PartialCellFront(grid; passes = 2)
PartialCellFront(passes = 2)
```
"""
function PartialCellFront(grid::AbstractGrid; passes = 4)
    passes ≥ 0 || throw(ArgumentError("passes must be non-negative, got $passes"))
    overflow = Field{Center, Center, Nothing}(grid)
    return PartialCellFront(overflow, passes)
end

Base.summary(front::PartialCellFront) = string("PartialCellFront(passes = ", front.passes, ")")
Base.show(io::IO, front::PartialCellFront) = print(io, summary(front))

#####
##### The full-cell flag
#####

initialize_front!(geometry, masking) = nothing

"""
$(TYPEDSIGNATURES)

Flag every cell holding ice as full, with no partial cells; called by `set!` when the thickness is given.
"""
function initialize_front!(geometry, ::PartialCellFront)
    (; H, ice_mask, front_thickness) = geometry
    grid = H.grid
    launch!(architecture(grid), grid, :xy, _initialize_full_cells!, ice_mask, front_thickness, H)
    return nothing
end

@kernel function _initialize_full_cells!(μ, Hʳ, H)
    i, j = @index(Global, NTuple)
    @inbounds begin
        μ[i, j, 1] = ifelse(H[i, j, 1] > 0, one(eltype(μ)), zero(eltype(μ)))
        Hʳ[i, j, 1] = zero(eltype(Hʳ))
    end
end

# The flag is prognostic: it changes only in `advance_front!`.
compute_ice_mask!(μ, ::PartialCellFront, H) = nothing

front_mask(masking, geometry) = nothing
front_mask(::PartialCellFront, geometry) = geometry.ice_mask

@inline full_cell(i, j, k, μ) = @inbounds μ[i, j, k] > 0
@inline full_neighbor(i, j, k, grid, μ) = full_cell(i, j, k, μ) & !inactive_cell(i, j, k, grid)
@inline empty_cell(i, j, k, grid, μ, H) = !full_cell(i, j, k, μ) & (@inbounds(H[i, j, k]) ≤ 0) & !inactive_cell(i, j, k, grid)

#####
##### Thickness fluxes: advection between full cells, upwind from a full cell into the front, none out of it
#####

@inline ice_source_mask(i, j, ::Nothing) = true
@inline ice_source_mask(i, j, μ) = full_cell(i, j, 1, μ)

@inline thickness_flux_divergence(i, j, k, grid, advection, U, H, ::Nothing) = div_Uc(i, j, k, grid, advection, U, H)

@inline function thickness_flux_divergence(i, j, k, grid, advection, U, H, μ)
    return (δxᶜᵃᵃ(i, j, k, grid, front_flux_x, advection, U.u, H, μ) +
            δyᵃᶜᵃ(i, j, k, grid, front_flux_y, advection, U.v, H, μ)) / Vᶜᶜᶜ(i, j, k, grid)
end

@inline function front_flux_x(i, j, k, grid, advection, u, H, μ)
    west = full_cell(i-1, j, k, μ)
    east = full_cell(i, j, k, μ)
    uᶠ = @inbounds u[i, j, k]
    Fᶠ = Axᶠᶜᶜ(i, j, k, grid) * ifelse(west, max(uᶠ, zero(uᶠ)) * @inbounds(H[i-1, j, k]), min(uᶠ, zero(uᶠ)) * @inbounds(H[i, j, k]))
    return ifelse(west & east, full_cell_flux_x(i, j, k, grid, advection, u, H), ifelse(west | east, Fᶠ, zero(grid)))
end

@inline function front_flux_y(i, j, k, grid, advection, v, H, μ)
    south = full_cell(i, j-1, k, μ)
    north = full_cell(i, j, k, μ)
    vᶠ = @inbounds v[i, j, k]
    Fᶠ = Ayᶜᶠᶜ(i, j, k, grid) * ifelse(south, max(vᶠ, zero(vᶠ)) * @inbounds(H[i, j-1, k]), min(vᶠ, zero(vᶠ)) * @inbounds(H[i, j, k]))
    return ifelse(south & north, full_cell_flux_y(i, j, k, grid, advection, v, H), ifelse(south | north, Fᶠ, zero(grid)))
end

@inline full_cell_flux_x(i, j, k, grid, advection, u, H) = _advective_tracer_flux_x(i, j, k, grid, advection, u, H)
@inline full_cell_flux_y(i, j, k, grid, advection, v, H) = _advective_tracer_flux_y(i, j, k, grid, advection, v, H)

# The bounds-preserving face flux, as in its `div_Uc`, evaluated one face at a time.
@inline function full_cell_flux_x(i, j, k, grid, advection::BoundsPreservingWENO, u, H)
    θ = advection.bounds.limiter
    Hᴸ = rescaled_reconstruction(_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, advection, LeftBias, H), i-1, j, k, grid, θ, H)
    Hᴿ = rescaled_reconstruction(_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, advection, RightBias, H), i, j, k, grid, θ, H)
    return Axᶠᶜᶜ(i, j, k, grid) * upwind_biased_product(@inbounds(u[i, j, k]), Hᴸ, Hᴿ)
end

@inline function full_cell_flux_y(i, j, k, grid, advection::BoundsPreservingWENO, v, H)
    θ = advection.bounds.limiter
    Hᴸ = rescaled_reconstruction(_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, advection, LeftBias, H), i, j-1, k, grid, θ, H)
    Hᴿ = rescaled_reconstruction(_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, advection, RightBias, H), i, j, k, grid, θ, H)
    return Ayᶜᶠᶜ(i, j, k, grid) * upwind_biased_product(@inbounds(v[i, j, k]), Hᴸ, Hᴿ)
end

#####
##### Front advance after the final stage
#####

advance_front!(masking, model, Δt) = nothing

"""
$(TYPEDSIGNATURES)

Empty the full cells that drained, update the ice thickness `Hʳ` of the front cells, fill the front cells that
reached it and pass their excess volume to empty neighbors, up to `front.passes` times, then apply the surface
mass balance and melt over `Δt` to the partial cells and to the cells that filled.
"""
function advance_front!(front::PartialCellFront, model, Δt)
    grid = model.grid
    arch = architecture(grid)
    (; H, ice_mask) = model.geometry
    (; u, v) = model.velocities
    Hʳ = model.geometry.front_thickness
    O = front.overflow

    fill_halo_regions!(H, model.clock, fields(model))
    launch!(arch, grid, :xy, _empty_drained_cells!, ice_mask, H)
    fill_halo_regions!(ice_mask)
    launch!(arch, grid, :xy, _compute_reference_thickness!, Hʳ, grid, ice_mask, H, u, v)

    for _ in 1:front.passes
        launch!(arch, grid, :xy, _fill_front_cells!, ice_mask, H, O, grid, Hʳ)
        fill_halo_regions!((ice_mask, O))
        fill_halo_regions!(H, model.clock, fields(model))
        launch!(arch, grid, :xy, _receive_overflow!, H, Hʳ, grid, ice_mask, O)
        fill_halo_regions!(H, model.clock, fields(model))
    end

    launch!(arch, grid, :xy, _force_front_cells!, H, Hʳ, ice_mask, model.geometry.clipped_thickness, model.surface_mass_balance, model.basal_melt_rate,
            floating_melt_fraction(model.grounding, model.geometry), coupled_grounded_melt(model.thermodynamics), Δt)
    fill_halo_regions!((ice_mask, Hʳ))
    fill_halo_regions!(H, model.clock, fields(model))

    return nothing
end

@kernel function _empty_drained_cells!(μ, H)
    i, j = @index(Global, NTuple)
    @inbounds μ[i, j, 1] = ifelse(H[i, j, 1] ≤ 0, zero(eltype(μ)), μ[i, j, 1])
end

@inline donor_thickness(i, j, k, grid, μ, H, inflow) = ifelse(inflow & full_neighbor(i, j, k, grid, μ), @inbounds(H[i, j, k]), zero(grid))
@inline donor_count(i, j, k, grid, μ, inflow) = ifelse(inflow & full_neighbor(i, j, k, grid, μ), 1, 0)

# Mean thickness of the full neighbors flowing in; otherwise the kept `Hʳ`, the mean of all full neighbors, or `H`.
@kernel function _compute_reference_thickness!(Hʳ, grid, μ, H, u, v)
    i, j = @index(Global, NTuple)
    k = 1
    @inbounds begin
        west = u[i, j, k] > 0
        east = u[i+1, j, k] < 0
        south = v[i, j, k] > 0
        north = v[i, j+1, k] < 0
    end
    Σⁱⁿ = donor_thickness(i-1, j, k, grid, μ, H, west) + donor_thickness(i+1, j, k, grid, μ, H, east) +
          donor_thickness(i, j-1, k, grid, μ, H, south) + donor_thickness(i, j+1, k, grid, μ, H, north)
    nⁱⁿ = donor_count(i-1, j, k, grid, μ, west) + donor_count(i+1, j, k, grid, μ, east) +
          donor_count(i, j-1, k, grid, μ, south) + donor_count(i, j+1, k, grid, μ, north)
    Σᶠ = donor_thickness(i-1, j, k, grid, μ, H, true) + donor_thickness(i+1, j, k, grid, μ, H, true) +
         donor_thickness(i, j-1, k, grid, μ, H, true) + donor_thickness(i, j+1, k, grid, μ, H, true)
    nᶠ = donor_count(i-1, j, k, grid, μ, true) + donor_count(i+1, j, k, grid, μ, true) +
         donor_count(i, j-1, k, grid, μ, true) + donor_count(i, j+1, k, grid, μ, true)
    @inbounds begin
        Hᵢ = H[i, j, k]
        Hˢ = ifelse(nᶠ > 0, Σᶠ / max(nᶠ, 1), Hᵢ)
        Hᵖ = ifelse(Hᵢ > 0, ifelse(Hʳ[i, j, k] > 0, Hʳ[i, j, k], Hˢ), zero(grid))
        H̄ = ifelse(nⁱⁿ > 0, Σⁱⁿ / max(nⁱⁿ, 1), Hᵖ)
        Hʳ[i, j, k] = ifelse(full_cell(i, j, k, μ), zero(grid), H̄)
    end
end

@inline empty_count(i, j, k, grid, μ, H) = ifelse(empty_cell(i, j, k, grid, μ, H), 1, 0)

# Candidates hold ice, so they never count as empty neighbors and the writes do not race.
@kernel function _fill_front_cells!(μ, H, O, grid, Hʳ)
    i, j = @index(Global, NTuple)
    k = 1
    n = empty_count(i-1, j, k, grid, μ, H) + empty_count(i+1, j, k, grid, μ, H) +
        empty_count(i, j-1, k, grid, μ, H) + empty_count(i, j+1, k, grid, μ, H)
    @inbounds begin
        Hᵢ = H[i, j, k]
        Hʳᵢ = Hʳ[i, j, k]
        fills = !full_cell(i, j, k, μ) & (Hʳᵢ > 0) & (Hᵢ ≥ Hʳᵢ)
        spills = fills & (n > 0)
        O[i, j, k] = ifelse(spills, (Hᵢ - Hʳᵢ) * Azᶜᶜᶜ(i, j, k, grid) / max(n, 1), zero(grid))
        H[i, j, k] = ifelse(spills, Hʳᵢ, Hᵢ)
        μ[i, j, k] = ifelse(fills, one(grid), μ[i, j, k])
    end
end

@inline received_volume(i, j, k, grid, O) = ifelse(inactive_cell(i, j, k, grid), zero(grid), @inbounds(O[i, j, k]))
@inline spill_donor(i, j, k, grid, O) = !inactive_cell(i, j, k, grid) & (@inbounds(O[i, j, k]) > 0)
@inline spilled_thickness(i, j, k, grid, H, O) = ifelse(spill_donor(i, j, k, grid, O), @inbounds(H[i, j, k]), zero(grid))
@inline spill_count(i, j, k, grid, O) = ifelse(spill_donor(i, j, k, grid, O), 1, 0)

# Every donor counted each of its empty neighbors, so the volume an empty cell gathers is all it was given.
@kernel function _receive_overflow!(H, Hʳ, grid, μ, O)
    i, j = @index(Global, NTuple)
    k = 1
    receives = empty_cell(i, j, k, grid, μ, H)
    V = received_volume(i-1, j, k, grid, O) + received_volume(i+1, j, k, grid, O) +
        received_volume(i, j-1, k, grid, O) + received_volume(i, j+1, k, grid, O)
    Σ = spilled_thickness(i-1, j, k, grid, H, O) + spilled_thickness(i+1, j, k, grid, H, O) +
        spilled_thickness(i, j-1, k, grid, H, O) + spilled_thickness(i, j+1, k, grid, H, O)
    n = spill_count(i-1, j, k, grid, O) + spill_count(i+1, j, k, grid, O) +
        spill_count(i, j-1, k, grid, O) + spill_count(i, j+1, k, grid, O)
    @inbounds begin
        H[i, j, k] = ifelse(receives, V / Azᶜᶜᶜ(i, j, k, grid), H[i, j, k])
        Hʳ[i, j, k] = ifelse(receives & (n > 0), Σ / max(n, 1), Hʳ[i, j, k])
    end
end

# Cells that filled this step still carry `Hʳ > 0`; their ice, like a partial cell's, missed the sources during the step.
@kernel function _force_front_cells!(H, Hʳ, μ, C, a, m, φ, mᵍ, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Hᵢ = H[i, j, 1]
        Hʳᵢ = Hʳ[i, j, 1]
        S = a[i, j, 1] - basal_melt(i, j, m, φ) - grounded_melt(i, j, mᵍ)
        full = full_cell(i, j, 1, μ)
        forced = (Hᵢ > 0) & (Hʳᵢ > 0)
        Hᶠ = ifelse(full, Hᵢ, Hʳᵢ)
        melted = forced & (Hᶠ + S * Δt ≤ 0)
        Hⁿ = ifelse(full, Hᵢ + S * Δt, Hᵢ / max(Hʳᵢ, eps(Hʳᵢ)) * (Hʳᵢ + S * Δt))
        C[i, j, 1] += ifelse(melted, -Hⁿ, zero(Hᵢ))
        H[i, j, 1] = ifelse(melted, zero(Hᵢ), ifelse(forced, Hⁿ, Hᵢ))
        Hʳ[i, j, 1] = ifelse(melted | full, zero(Hʳᵢ), ifelse(forced, Hʳᵢ + S * Δt, Hʳᵢ))
        μ[i, j, 1] = ifelse(melted, zero(eltype(μ)), μ[i, j, 1])
    end
end
