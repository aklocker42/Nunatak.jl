"""
$(TYPEDEF)

Geometric multigrid V-cycle preconditioner for `ssa_operator!`. Each level halves `Nx` and `Ny` and rediscretizes
the operator from the area-weighted average of `η̄H` and the restricted drag; `η̄Hᶠᶠ` is interpolated to the coarse
corners as in the Picard coefficient update. Velocities are transferred on the C grid, linearly along each
component's own direction and with 3/4–1/4 weights across it, and restriction is the transpose of prolongation.
The smoother is ℓ1-Jacobi, whose diagonal bounds the absolute row sum of the operator, so every sweep is
convergent and the V-cycle is symmetric positive definite, as conjugate gradients require. Wall and ice-free
velocity points are excluded from the transfers. The coarse coefficients are rebuilt at the start of every
conjugate-gradient solve, so the preconditioner follows the coefficients through Picard iterations.

$(TYPEDFIELDS)
"""
struct StressBalanceMultigridPreconditioner{L, W}
    "coarse levels, finest first, each a `NamedTuple` of grid, coefficients and work vectors"
    levels :: L
    "residual workspace on the finest grid"
    residual :: W
    "ℓ1-Jacobi sweeps before and after each coarse-grid correction"
    smoothing_sweeps :: Int
    "ℓ1-Jacobi sweeps on the coarsest grid"
    coarsest_sweeps :: Int
end

"""
$(TYPEDSIGNATURES)

Build a `StressBalanceMultigridPreconditioner` on `grid`. The grid is halved while both `Nx` and `Ny` are even and
the coarse grid keeps at least `minimum_coarse_size` cells, and no fewer cells than halo points, in each direction.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (32, 16), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat));

julia> P = StressBalanceMultigridPreconditioner(grid);

julia> map(level -> size(level.grid), P.levels)
((16, 8, 1), (8, 4, 1))
```
"""
function StressBalanceMultigridPreconditioner(grid; smoothing_sweeps = 2, coarsest_sweeps = 20, minimum_coarse_size = 2)
    levels = coarse_levels(grid, minimum_coarse_size)
    isempty(levels) && @warn "$(size(grid)) grid cannot be halved; the multigrid preconditioner reduces to ℓ1-Jacobi sweeps"
    return StressBalanceMultigridPreconditioner(levels, VelocityVector(grid), smoothing_sweeps, coarsest_sweeps)
end

function coarse_levels(grid, minimum_coarse_size)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    coarsenable = iseven(Nx) & iseven(Ny) & (Nx ÷ 2 ≥ max(minimum_coarse_size, Hx)) & (Ny ÷ 2 ≥ max(minimum_coarse_size, Hy))
    coarsenable || return ()
    coarse = coarsened_grid(grid)
    level = (grid = coarse,
             η̄H = Field{Center, Center, Nothing}(coarse),
             η̄Hᶠᶠ = Field{Face, Face, Nothing}(coarse),
             βᵘ = Field{Face, Center, Nothing}(coarse),
             βᵛ = Field{Center, Face, Nothing}(coarse),
             x = VelocityVector(coarse),
             b = VelocityVector(coarse),
             q = VelocityVector(coarse))
    return (level, coarse_levels(coarse, minimum_coarse_size)...)
end

function coarsened_grid(grid)
    cpu_grid = on_architecture(CPU(), grid)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    TX, TY, _ = topology(grid)
    x = [xnode(i, cpu_grid, Face()) for i in 1:2:Nx+1]
    y = [ynode(j, cpu_grid, Face()) for j in 1:2:Ny+1]
    return RectilinearGrid(architecture(grid), eltype(grid); size = (Nx ÷ 2, Ny ÷ 2), halo = (Hx, Hy), x, y, topology = (TX, TY, Flat))
end

function precondition!(z::VelocityVector, P::StressBalanceMultigridPreconditioner, r::VelocityVector, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, args...)
    walls = side_walls(args...)
    v_cycle!(z, r, P.residual, (; η̄H, η̄Hᶠᶠ, βᵘ, βᵛ), P.levels, walls, P)
    return nothing
end

update_preconditioner!(P::StressBalanceMultigridPreconditioner, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, args...) = coarsen_coefficients!(P.levels, η̄H, βᵘ, βᵛ)

coarsen_coefficients!(::Tuple{}, η̄H, βᵘ, βᵛ) = nothing

function coarsen_coefficients!(levels::Tuple, η̄H, βᵘ, βᵛ)
    level = first(levels)
    grid = level.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _coarsen_coefficients!, level.η̄H, level.βᵘ, level.βᵛ, η̄H, βᵘ, βᵛ, grid, η̄H.grid)
    fill_halo_regions!(level.η̄H)
    launch!(arch, grid, size(level.η̄Hᶠᶠ), _interpolate_to_corners!, level.η̄Hᶠᶠ, level.η̄H, grid)
    fill_halo_regions!(level.η̄Hᶠᶠ)
    return coarsen_coefficients!(Base.tail(levels), level.η̄H, level.βᵘ, level.βᵛ)
end

# `x ≈ A⁻¹ b` from a zero initial guess; `q` is workspace on the same grid.
function v_cycle!(x, b, q, coefficients, ::Tuple{}, walls, P)
    fill!(x, 0)
    smooth!(x, b, q, coefficients, walls, P.coarsest_sweeps)
    return nothing
end

function v_cycle!(x, b, q, coefficients, levels::Tuple, walls, P)
    coarse = first(levels)
    fill!(x, 0)
    smooth!(x, b, q, coefficients, walls, P.smoothing_sweeps)
    fill_halo_regions!(x)
    ssa_operator!(q, x, coefficients.η̄H, coefficients.η̄Hᶠᶠ, coefficients.βᵘ, coefficients.βᵛ, walls)
    subtract!(q, b, q)
    restrict!(coarse.b, q, coarse.η̄H, coefficients.η̄H)
    coarse_coefficients = (η̄H = coarse.η̄H, η̄Hᶠᶠ = coarse.η̄Hᶠᶠ, βᵘ = coarse.βᵘ, βᵛ = coarse.βᵛ)
    v_cycle!(coarse.x, coarse.b, coarse.q, coarse_coefficients, Base.tail(levels), walls, P)
    prolongate!(x, coarse.x, coefficients.η̄H, coarse.η̄H)
    smooth!(x, b, q, coefficients, walls, P.smoothing_sweeps)
    return nothing
end

function smooth!(x, b, q, coefficients, walls, sweeps)
    (; η̄H, η̄Hᶠᶠ, βᵘ, βᵛ) = coefficients
    grid = η̄H.grid
    for _ in 1:sweeps
        fill_halo_regions!(x)
        ssa_operator!(q, x, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls)
        launch!(architecture(grid), grid, :xyz, _smooth!, x.u, x.v, b.u, b.v, q.u, q.v, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid)
    end
    return nothing
end

function restrict!(coarse, fine, coarse_η̄H, fine_η̄H)
    grid = coarse_η̄H.grid
    launch!(architecture(grid), grid, :xyz, _restrict!, coarse.u, coarse.v, fine.u, fine.v, coarse_η̄H, fine_η̄H, grid, fine_η̄H.grid)
    return nothing
end

function prolongate!(fine, coarse, fine_η̄H, coarse_η̄H)
    grid = fine_η̄H.grid
    launch!(architecture(grid), grid, :xyz, _prolongate!, fine.u, fine.v, coarse.u, coarse.v, fine_η̄H, coarse_η̄H, grid, coarse_η̄H.grid)
    return nothing
end

# ℓ1-Jacobi diagonal: absolute row sum, so `M ≥ A` and the sweep converges undamped.
@inline function x_absolute_row_sum(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵘ, walls)
    s = @inbounds Azᶠᶜᶜ(i, j, k, grid) * βᵘ[i, j, k] +
        2 * (x_normal_diagonal(i, j, k, grid, η̄H) + x_normal_diagonal(i-1, j, k, grid, η̄H)) +
        4 * (η̄H[i, j, k] + η̄H[i-1, j, k]) +
        2 * (x_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ, walls) + x_shear_diagonal(i, j+1, k, grid, η̄Hᶠᶠ, walls)) +
        2 * (corner_shear_factor(i, j, k, grid, walls) * η̄Hᶠᶠ[i, j, k] + corner_shear_factor(i, j+1, k, grid, walls) * η̄Hᶠᶠ[i, j+1, k])
    return ifelse(inactiveᶠᶜᶜ(i, j, k, grid, η̄H) | iszero(s), one(grid), s)
end

@inline function y_absolute_row_sum(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵛ, walls)
    s = @inbounds Azᶜᶠᶜ(i, j, k, grid) * βᵛ[i, j, k] +
        2 * (y_normal_diagonal(i, j, k, grid, η̄H) + y_normal_diagonal(i, j-1, k, grid, η̄H)) +
        4 * (η̄H[i, j, k] + η̄H[i, j-1, k]) +
        2 * (y_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ, walls) + y_shear_diagonal(i+1, j, k, grid, η̄Hᶠᶠ, walls)) +
        2 * (corner_shear_factor(i, j, k, grid, walls) * η̄Hᶠᶠ[i, j, k] + corner_shear_factor(i+1, j, k, grid, walls) * η̄Hᶠᶠ[i+1, j, k])
    return ifelse(inactiveᶜᶠᶜ(i, j, k, grid, η̄H) | iszero(s), one(grid), s)
end

@kernel function _smooth!(xu, xv, bu, bv, qu, qv, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        xu[i, j, k] += (bu[i, j, k] - qu[i, j, k]) / x_absolute_row_sum(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵘ, walls)
        xv[i, j, k] += (bv[i, j, k] - qv[i, j, k]) / y_absolute_row_sum(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵛ, walls)
    end
end

# Fine faces `2I - 1 + a`, `a ∈ -1:1`, restrict onto coarse face `I` with weights 1/2, 1, 1/2; fine centers
# `2J - 1 + b`, `b ∈ -1:2`, onto coarse center `J` with weights 1/4, 3/4, 3/4, 1/4, folded at `Bounded` edges.
@inline restriction_offsets(::Face) = -1:1
@inline restriction_offsets(::Center) = -1:2

@inline restriction_index(I, a, N, ::Periodic, ℓ) = mod1(2I - 1 + a, N)
@inline restriction_index(I, a, N, ::Bounded, ::Face) = max(2I - 1 + a, 1)
@inline restriction_index(I, a, N, ::Bounded, ::Center) = clamp(2I - 1 + a, 1, N)

@inline restriction_weight(I, a, Nc, ::Periodic, ::Face, FT) = ifelse(a == 0, one(FT), one(FT) / 2)
@inline restriction_weight(I, a, Nc, ::Bounded, ::Face, FT) = ifelse(2I - 1 + a < 1, zero(FT), restriction_weight(I, a, Nc, Periodic(), Face(), FT))
@inline restriction_weight(I, a, Nc, ::Periodic, ::Center, FT) = ifelse((a == 0) | (a == 1), 3 * one(FT) / 4, one(FT) / 4)

@inline function restriction_weight(I, a, Nc, ::Bounded, ::Center, FT)
    outside = ((I == 1) & (a == -1)) | ((I == Nc) & (a == 2))
    folded = ((I == 1) & (a == 0)) | ((I == Nc) & (a == 1))
    w = restriction_weight(I, a, Nc, Periodic(), Center(), FT) + ifelse(folded, one(FT) / 4, zero(FT))
    return ifelse(outside, zero(FT), w)
end

# Coarse parents of fine point `i` with their weights, the transpose of the restriction.
@inline function parents(i, Nc, topology, ::Face, FT)
    I = (i + 1) ÷ 2
    odd = isodd(i)
    I⁺ = ifelse(odd, I, upper_face(I, Nc, topology))
    return I, I⁺, ifelse(odd, one(FT), one(FT) / 2), ifelse(odd, zero(FT), one(FT) / 2)
end

@inline function parents(i, Nc, topology, ::Center, FT)
    I = (i + 1) ÷ 2
    I′ = ifelse(isodd(i), I - 1, I + 1)
    return I, adjacent_center(I′, Nc, topology), 3 * one(FT) / 4, one(FT) / 4
end

@inline upper_face(I, Nc, ::Periodic) = mod1(I + 1, Nc)
@inline upper_face(I, Nc, ::Bounded) = I + 1

@inline adjacent_center(I, Nc, ::Periodic) = mod1(I, Nc)
@inline adjacent_center(I, Nc, ::Bounded) = clamp(I, 1, Nc)

@inline function restrict(I, J, k, ℓx, ℓy, coarse_grid, fine_grid, ϕ, args...)
    TX, TY, _ = topology(fine_grid)
    FT = eltype(fine_grid)
    s = zero(FT)
    for a in restriction_offsets(ℓx), b in restriction_offsets(ℓy)
        i = restriction_index(I, a, fine_grid.Nx, TX(), ℓx)
        j = restriction_index(J, b, fine_grid.Ny, TY(), ℓy)
        w = restriction_weight(I, a, coarse_grid.Nx, TX(), ℓx, FT) * restriction_weight(J, b, coarse_grid.Ny, TY(), ℓy, FT)
        s += w * ϕ(i, j, k, fine_grid, args...)
    end
    return s
end

@inline function prolongate(i, j, k, ℓx, ℓy, fine_grid, coarse_grid, ϕ, args...)
    TX, TY, _ = topology(fine_grid)
    FT = eltype(fine_grid)
    I₁, I₂, wx₁, wx₂ = parents(i, coarse_grid.Nx, TX(), ℓx, FT)
    J₁, J₂, wy₁, wy₂ = parents(j, coarse_grid.Ny, TY(), ℓy, FT)
    return wx₁ * (wy₁ * ϕ(I₁, J₁, k, coarse_grid, args...) + wy₂ * ϕ(I₁, J₂, k, coarse_grid, args...)) +
           wx₂ * (wy₁ * ϕ(I₂, J₁, k, coarse_grid, args...) + wy₂ * ϕ(I₂, J₂, k, coarse_grid, args...))
end

@inline area_weighted(i, j, k, grid, ϕ, ℓx, ℓy) = @inbounds Az(i, j, k, grid, ℓx, ℓy, Center()) * ϕ[i, j, k]

@kernel function _coarsen_coefficients!(η̄H, βᵘ, βᵛ, fine_η̄H, fine_βᵘ, fine_βᵛ, grid, fine_grid)
    I, J, k = @index(Global, NTuple)
    i, j = 2I - 1, 2J - 1
    @inbounds begin
        Σ = zero(grid)
        for a in 0:1, b in 0:1
            Σ += area_weighted(i + a, j + b, k, fine_grid, fine_η̄H, Center(), Center())
        end
        η̄H[I, J, k] = Σ / Azᶜᶜᶜ(I, J, k, grid)
        βᵘ[I, J, k] = restrict(I, J, k, Face(), Center(), grid, fine_grid, area_weighted, fine_βᵘ, Face(), Center()) / Azᶠᶜᶜ(I, J, k, grid)
        βᵛ[I, J, k] = restrict(I, J, k, Center(), Face(), grid, fine_grid, area_weighted, fine_βᵛ, Center(), Face()) / Azᶜᶠᶜ(I, J, k, grid)
    end
end

@kernel function _restrict!(bu, bv, ru, rv, η̄H, fine_η̄H, grid, fine_grid)
    I, J, k = @index(Global, NTuple)
    @inbounds begin
        bu[I, J, k] = ifelse(inactiveᶠᶜᶜ(I, J, k, grid, η̄H), zero(grid), restrict(I, J, k, Face(), Center(), grid, fine_grid, interior_u, ru, fine_η̄H))
        bv[I, J, k] = ifelse(inactiveᶜᶠᶜ(I, J, k, grid, η̄H), zero(grid), restrict(I, J, k, Center(), Face(), grid, fine_grid, interior_v, rv, fine_η̄H))
    end
end

@kernel function _prolongate!(xu, xv, cu, cv, η̄H, coarse_η̄H, grid, coarse_grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        xu[i, j, k] += ifelse(inactiveᶠᶜᶜ(i, j, k, grid, η̄H), zero(grid), prolongate(i, j, k, Face(), Center(), grid, coarse_grid, interior_u, cu, coarse_η̄H))
        xv[i, j, k] += ifelse(inactiveᶜᶠᶜ(i, j, k, grid, η̄H), zero(grid), prolongate(i, j, k, Center(), Face(), grid, coarse_grid, interior_v, cv, coarse_η̄H))
    end
end
