"""
$(TYPEDEF)

Geometric multigrid V-cycle preconditioner for `ssa_operator!`. Each level halves `Nx`, `Ny` or both, keeping a
direction unchanged once it can no longer be halved (semi-coarsening), and rediscretizes the operator from the
area-weighted average of `η̄H` and the restricted drag; `η̄Hᶠᶠ` is interpolated to the coarse corners as in the
Picard coefficient update. Velocities are transferred on the C grid, linearly along each component's own direction
and with 3/4–1/4 weights across it, and restriction is the transpose of prolongation.
The smoother is ℓ1-Jacobi on points or, by default, on lines in `y`: each sweep solves the tridiagonal block of
every line and adds the absolute couplings that leave it to the diagonal, so every sweep is convergent and the
V-cycle is symmetric positive definite, as conjugate gradients require. Lines resolve the strong `y` coupling of
cells stretched in `x` by semi-coarsening. Wall and ice-free
velocity points are excluded from the transfers. The coarse coefficients are rebuilt at the start of every
conjugate-gradient solve, so the preconditioner follows the coefficients through Picard iterations.

$(TYPEDFIELDS)
"""
struct StressBalanceMultigridPreconditioner{L, W, T, S}
    "coarse levels, finest first, each a `NamedTuple` of grid, coefficients and work vectors"
    levels :: L
    "residual workspace on the finest grid"
    residual :: W
    "line-solve workspace on the finest grid"
    scratch :: T
    "smoother, `PointJacobi()` or `LineJacobi()`"
    smoother :: S
    "smoothing sweeps before and after each coarse-grid correction"
    smoothing_sweeps :: Int
    "smoothing sweeps on the coarsest grid"
    coarsest_sweeps :: Int
end

"""
$(TYPEDSIGNATURES)

Build a `StressBalanceMultigridPreconditioner` on `grid`. A direction is halved while its size is even and the coarse
grid keeps at least `minimum_coarse_size` cells, and no fewer cells than halo points, in that direction; coarsening
stops when neither direction can be halved. On a distributed grid every rank halves its own part, so each coarse
grid is partitioned like `grid`; partitions must be equal. `smoother` is `PointJacobi()` or `LineJacobi()`; use `LineJacobi()` on
grids much longer in `x` than in `y`, such as flowlines.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (32, 16), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat));

julia> P = StressBalanceMultigridPreconditioner(grid);

julia> map(level -> size(level.grid), P.levels)
((16, 8, 1), (8, 4, 1), (4, 4, 1))

julia> flowline = RectilinearGrid(size = (64, 4), halo = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat));

julia> map(level -> size(level.grid), StressBalanceMultigridPreconditioner(flowline).levels)
((32, 4, 1), (16, 4, 1), (8, 4, 1), (4, 4, 1))
```
"""
function StressBalanceMultigridPreconditioner(grid; smoother = PointJacobi(), smoothing_sweeps = 2, coarsest_sweeps = 20, minimum_coarse_size = 2)
    levels = coarse_levels(grid, minimum_coarse_size)
    isempty(levels) && @warn "$(size(grid)) grid cannot be halved; the multigrid preconditioner reduces to ℓ1-Jacobi sweeps"
    scratch = Field{Center, Center, Nothing}(grid)
    return StressBalanceMultigridPreconditioner(levels, VelocityVector(grid), scratch, smoother, smoothing_sweeps, coarsest_sweeps)
end

"""
$(TYPEDEF)

ℓ1-Jacobi smoother on single velocity points for `StressBalanceMultigridPreconditioner`.
"""
struct PointJacobi end

"""
$(TYPEDEF)

ℓ1-block Jacobi smoother on lines in `y` for `StressBalanceMultigridPreconditioner`. It needs fewer iterations than
[`PointJacobi`](@ref) when the cells are much longer in `x` than in `y`, as on the coarse levels of a flowline. Each
line is solved sequentially, so a sweep costs more, most of all on a GPU.
"""
struct LineJacobi end

struct Halved end
struct Unchanged end

coarsening_factor(::Halved) = 2
coarsening_factor(::Unchanged) = 1

# A direction is halved only if every rank can halve its part.
function coarsening(N, H, minimum_coarse_size, arch)
    halvable = iseven(N) & (N ÷ 2 ≥ max(minimum_coarse_size, H))
    return ifelse(all_reduce(min, Int(halvable), arch) == 1, Halved(), Unchanged())
end

function coarse_levels(grid, minimum_coarse_size)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    arch = architecture(grid)
    return coarse_levels(grid, coarsening(Nx, Hx, minimum_coarse_size, arch), coarsening(Ny, Hy, minimum_coarse_size, arch), minimum_coarse_size)
end

coarse_levels(grid, ::Unchanged, ::Unchanged, minimum_coarse_size) = ()

function coarse_levels(grid, cx, cy, minimum_coarse_size)
    coarse = coarsened_grid(grid, cx, cy)
    level = (grid = coarse,
             coarsening = (cx, cy),
             η̄H = Field{Center, Center, Nothing}(coarse),
             η̄Hᶠᶠ = Field{Face, Face, Nothing}(coarse),
             βᵘ = Field{Face, Center, Nothing}(coarse),
             βᵛ = Field{Center, Face, Nothing}(coarse),
             x = VelocityVector(coarse),
             b = VelocityVector(coarse),
             q = VelocityVector(coarse),
             scratch = Field{Center, Center, Nothing}(coarse))
    return (level, coarse_levels(coarse, minimum_coarse_size)...)
end

function coarsened_grid(grid, cx, cy)
    global_grid = on_architecture(CPU(), reconstruct_global_grid(grid))
    Nx, Ny, _ = size(global_grid)
    Hx, Hy, _ = halo_size(grid)
    TX, TY, _ = topology(global_grid)
    fx, fy = coarsening_factor(cx), coarsening_factor(cy)
    x = [xnode(i, global_grid, Face()) for i in 1:fx:Nx+1]
    y = [ynode(j, global_grid, Face()) for j in 1:fy:Ny+1]
    coarse = RectilinearGrid(architecture(grid), eltype(grid); size = (Nx ÷ fx, Ny ÷ fy), halo = (Hx, Hy), x, y, topology = (TX, TY, Flat))
    size(coarse, 1) * fx == size(grid, 1) && size(coarse, 2) * fy == size(grid, 2) ||
        throw(ArgumentError("the coarse grid is not partitioned like $(size(grid)) grid; use equal partitions"))
    return coarse
end

function precondition!(z::VelocityVector, P::StressBalanceMultigridPreconditioner, r::VelocityVector, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, args...)
    walls = side_walls(args...)
    v_cycle!(z, r, P.residual, P.scratch, (; η̄H, η̄Hᶠᶠ, βᵘ, βᵛ), P.levels, walls, P)
    return nothing
end

update_preconditioner!(P::StressBalanceMultigridPreconditioner, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, args...) = coarsen_coefficients!(P.levels, η̄H, βᵘ, βᵛ)

coarsen_coefficients!(::Tuple{}, η̄H, βᵘ, βᵛ) = nothing

function coarsen_coefficients!(levels::Tuple, η̄H, βᵘ, βᵛ)
    level = first(levels)
    grid = level.grid
    arch = architecture(grid)
    fill_halo_regions!((βᵘ, βᵛ))
    launch!(arch, grid, :xyz, _coarsen_coefficients!, level.η̄H, level.βᵘ, level.βᵛ, η̄H, βᵘ, βᵛ, level.coarsening..., grid, η̄H.grid)
    fill_halo_regions!(level.η̄H)
    launch!(arch, grid, size(level.η̄Hᶠᶠ), _interpolate_to_corners!, level.η̄Hᶠᶠ, level.η̄H, grid)
    fill_halo_regions!(level.η̄Hᶠᶠ)
    return coarsen_coefficients!(Base.tail(levels), level.η̄H, level.βᵘ, level.βᵛ)
end

# `x ≈ A⁻¹ b` from a zero initial guess; `q` and `t` are workspace on the same grid.
function v_cycle!(x, b, q, t, coefficients, ::Tuple{}, walls, P)
    fill!(x, 0)
    smooth!(x, b, q, t, coefficients, walls, P.smoother, P.coarsest_sweeps)
    return nothing
end

function v_cycle!(x, b, q, t, coefficients, levels::Tuple, walls, P)
    coarse = first(levels)
    fill!(x, 0)
    smooth!(x, b, q, t, coefficients, walls, P.smoother, P.smoothing_sweeps)
    fill_halo_regions!(x)
    ssa_operator!(q, x, coefficients.η̄H, coefficients.η̄Hᶠᶠ, coefficients.βᵘ, coefficients.βᵛ, walls)
    subtract!(q, b, q)
    fill_halo_regions!(q)
    restrict!(coarse.b, q, coarse.η̄H, coefficients.η̄H, coarse.coarsening)
    coarse_coefficients = (η̄H = coarse.η̄H, η̄Hᶠᶠ = coarse.η̄Hᶠᶠ, βᵘ = coarse.βᵘ, βᵛ = coarse.βᵛ)
    v_cycle!(coarse.x, coarse.b, coarse.q, coarse.scratch, coarse_coefficients, Base.tail(levels), walls, P)
    fill_halo_regions!(coarse.x)
    prolongate!(x, coarse.x, coefficients.η̄H, coarse.η̄H, coarse.coarsening)
    smooth!(x, b, q, t, coefficients, walls, P.smoother, P.smoothing_sweeps)
    return nothing
end

function smooth!(x, b, q, t, coefficients, walls, smoother, sweeps)
    (; η̄H, η̄Hᶠᶠ, βᵘ, βᵛ) = coefficients
    grid = η̄H.grid
    for _ in 1:sweeps
        fill_halo_regions!(x)
        ssa_operator!(q, x, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls)
        relax!(x, b, q, t, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid, smoother)
    end
    return nothing
end

relax!(x, b, q, t, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid, ::PointJacobi) =
    launch!(architecture(grid), grid, :xyz, _smooth!, x.u, x.v, b.u, b.v, q.u, q.v, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid)

relax!(x, b, q, t, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid, ::LineJacobi) =
    launch!(architecture(grid), grid, :xz, _line_smooth!, x.u, x.v, b.u, b.v, q.u, q.v, t, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid)

function restrict!(coarse, fine, coarse_η̄H, fine_η̄H, coarsening)
    grid = coarse_η̄H.grid
    launch!(architecture(grid), grid, :xyz, _restrict!, coarse.u, coarse.v, fine.u, fine.v, coarse_η̄H, fine_η̄H, coarsening..., grid, fine_η̄H.grid)
    return nothing
end

function prolongate!(fine, coarse, fine_η̄H, coarse_η̄H, coarsening)
    grid = fine_η̄H.grid
    launch!(architecture(grid), grid, :xyz, _prolongate!, fine.u, fine.v, coarse.u, coarse.v, fine_η̄H, coarse_η̄H, coarsening..., grid, coarse_η̄H.grid)
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

# ℓ1-block Jacobi on y lines: the tridiagonal block of each line plus, on its diagonal, the absolute couplings
# leaving the line, so `M ≥ A` as for point ℓ1-Jacobi. Couplings to inactive points are zero in the operator.
@inline x_line_coupling(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵘ, walls) =
    ifelse(inactiveᶠᶜᶜ(i, j-1, k, grid, η̄H) | inactiveᶠᶜᶜ(i, j, k, grid, η̄H), zero(grid), x_shear_diagonal(i, j, k, grid, η̄Hᶠᶠ, walls))

@inline y_line_coupling(i, j, k, grid, η̄H, η̄Hᶠᶠ, βᵛ, walls) =
    ifelse(inactiveᶜᶠᶜ(i, j-1, k, grid, η̄H) | inactiveᶜᶠᶜ(i, j, k, grid, η̄H), zero(grid), y_normal_diagonal(i, j-1, k, grid, η̄H))

# Thomas algorithm for `M δ = b - q` along `j`, with diagonal `s - cⱼ - cⱼ₊₁` and off-diagonals `-c`;
# `t` keeps the eliminated upper diagonal and `q` the eliminated right-hand side.
@inline function line_relax!(x, b, q, t, i, k, grid, absolute_row_sum, line_coupling, coefficients...)
    Ny = size(grid, 2)
    FT = eltype(grid)
    cⱼ = zero(FT)
    tⱼ = zero(FT)
    dⱼ = zero(FT)
    for j in 1:Ny
        s = absolute_row_sum(i, j, k, grid, coefficients...)
        cⱼ₊₁ = ifelse(j == Ny, zero(FT), line_coupling(i, j+1, k, grid, coefficients...))
        m = s - cⱼ - cⱼ₊₁ + cⱼ * tⱼ
        @inbounds dⱼ = (b[i, j, k] - q[i, j, k] + cⱼ * dⱼ) / m
        tⱼ = -cⱼ₊₁ / m
        @inbounds t[i, j, k] = tⱼ
        @inbounds q[i, j, k] = dⱼ
        cⱼ = cⱼ₊₁
    end
    δ = zero(FT)
    for j in Ny:-1:1
        @inbounds δ = q[i, j, k] - t[i, j, k] * δ
        @inbounds x[i, j, k] += δ
    end
    return nothing
end

@kernel function _line_smooth!(xu, xv, bu, bv, qu, qv, t, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, walls, grid)
    i, k = @index(Global, NTuple)
    line_relax!(xu, bu, qu, t, i, k, grid, x_absolute_row_sum, x_line_coupling, η̄H, η̄Hᶠᶠ, βᵘ, walls)
    line_relax!(xv, bv, qv, t, i, k, grid, y_absolute_row_sum, y_line_coupling, η̄H, η̄Hᶠᶠ, βᵛ, walls)
end

# Fine faces `2I - 1 + a`, `a ∈ -1:1`, restrict onto coarse face `I` with weights 1/2, 1, 1/2; fine centers
# `2J - 1 + b`, `b ∈ -1:2`, onto coarse center `J` with weights 1/4, 3/4, 3/4, 1/4, folded at `Closed` sides.
# Across a `Connected` side the stencil reads the filled halo. An `Unchanged` direction transfers point to point.
@inline restriction_offsets(::Face, ::Halved) = -1:1
@inline restriction_offsets(::Center, ::Halved) = -1:2
@inline restriction_offsets(ℓ, ::Unchanged) = 0:0

@inline closed_left(::TwoSided{Closed}) = true
@inline closed_left(topology) = false
@inline closed_right(::TwoSided{<:Any, Closed}) = true
@inline closed_right(topology) = false

@inline restriction_index(I, a, N, ::Periodic, ℓ, ::Halved) = mod1(2I - 1 + a, N)
@inline restriction_index(I, a, N, topology::TwoSided, ::Face, ::Halved) = ifelse(closed_left(topology), max(2I - 1 + a, 1), 2I - 1 + a)
@inline restriction_index(I, a, N, topology::TwoSided, ::Center, ::Halved) =
    clamp(2I - 1 + a, ifelse(closed_left(topology), 1, 0), ifelse(closed_right(topology), N, N + 1))
@inline restriction_index(I, a, N, topology, ℓ, ::Unchanged) = I

@inline restriction_weight(I, a, Nc, ::Periodic, ::Face, ::Halved, FT) = ifelse(a == 0, one(FT), one(FT) / 2)
@inline restriction_weight(I, a, Nc, topology::TwoSided, ::Face, ::Halved, FT) =
    ifelse(closed_left(topology) & (2I - 1 + a < 1), zero(FT), restriction_weight(I, a, Nc, Periodic(), Face(), Halved(), FT))
@inline restriction_weight(I, a, Nc, ::Periodic, ::Center, ::Halved, FT) = ifelse((a == 0) | (a == 1), 3 * one(FT) / 4, one(FT) / 4)
@inline restriction_weight(I, a, Nc, topology, ℓ, ::Unchanged, FT) = one(FT)

@inline function restriction_weight(I, a, Nc, topology::TwoSided, ::Center, ::Halved, FT)
    left, right = closed_left(topology) & (I == 1), closed_right(topology) & (I == Nc)
    outside = (left & (a == -1)) | (right & (a == 2))
    folded = (left & (a == 0)) | (right & (a == 1))
    w = restriction_weight(I, a, Nc, Periodic(), Center(), Halved(), FT) + ifelse(folded, one(FT) / 4, zero(FT))
    return ifelse(outside, zero(FT), w)
end

# Coarse parents of fine point `i` with their weights, the transpose of the restriction.
@inline function parents(i, Nc, topology, ::Face, ::Halved, FT)
    I = (i + 1) ÷ 2
    odd = isodd(i)
    I⁺ = ifelse(odd, I, upper_face(I, Nc, topology))
    return I, I⁺, ifelse(odd, one(FT), one(FT) / 2), ifelse(odd, zero(FT), one(FT) / 2)
end

@inline function parents(i, Nc, topology, ::Center, ::Halved, FT)
    I = (i + 1) ÷ 2
    I′ = ifelse(isodd(i), I - 1, I + 1)
    return I, adjacent_center(I′, Nc, topology), 3 * one(FT) / 4, one(FT) / 4
end

@inline parents(i, Nc, topology, ℓ, ::Unchanged, FT) = i, i, one(FT), zero(FT)

@inline upper_face(I, Nc, ::Periodic) = mod1(I + 1, Nc)
@inline upper_face(I, Nc, ::TwoSided) = I + 1

@inline adjacent_center(I, Nc, ::Periodic) = mod1(I, Nc)
@inline adjacent_center(I, Nc, topology::TwoSided) = clamp(I, ifelse(closed_left(topology), 1, 0), ifelse(closed_right(topology), Nc, Nc + 1))

@inline function restrict(I, J, k, ℓx, ℓy, cx, cy, coarse_grid, fine_grid, ϕ, args...)
    TX, TY, _ = topology(fine_grid)
    FT = eltype(fine_grid)
    s = zero(FT)
    for a in restriction_offsets(ℓx, cx), b in restriction_offsets(ℓy, cy)
        i = restriction_index(I, a, fine_grid.Nx, TX(), ℓx, cx)
        j = restriction_index(J, b, fine_grid.Ny, TY(), ℓy, cy)
        w = restriction_weight(I, a, coarse_grid.Nx, TX(), ℓx, cx, FT) * restriction_weight(J, b, coarse_grid.Ny, TY(), ℓy, cy, FT)
        s += w * ϕ(i, j, k, fine_grid, args...)
    end
    return s
end

@inline function prolongate(i, j, k, ℓx, ℓy, cx, cy, fine_grid, coarse_grid, ϕ, args...)
    TX, TY, _ = topology(fine_grid)
    FT = eltype(fine_grid)
    I₁, I₂, wx₁, wx₂ = parents(i, coarse_grid.Nx, TX(), ℓx, cx, FT)
    J₁, J₂, wy₁, wy₂ = parents(j, coarse_grid.Ny, TY(), ℓy, cy, FT)
    return wx₁ * (wy₁ * ϕ(I₁, J₁, k, coarse_grid, args...) + wy₂ * ϕ(I₁, J₂, k, coarse_grid, args...)) +
           wx₂ * (wy₁ * ϕ(I₂, J₁, k, coarse_grid, args...) + wy₂ * ϕ(I₂, J₂, k, coarse_grid, args...))
end

@inline area_weighted(i, j, k, grid, ϕ, ℓx, ℓy) = @inbounds Az(i, j, k, grid, ℓx, ℓy, Center()) * ϕ[i, j, k]

@kernel function _coarsen_coefficients!(η̄H, βᵘ, βᵛ, fine_η̄H, fine_βᵘ, fine_βᵛ, cx, cy, grid, fine_grid)
    I, J, k = @index(Global, NTuple)
    fx, fy = coarsening_factor(cx), coarsening_factor(cy)
    i, j = fx * (I - 1) + 1, fy * (J - 1) + 1
    @inbounds begin
        Σ = zero(grid)
        for a in 0:fx-1, b in 0:fy-1
            Σ += area_weighted(i + a, j + b, k, fine_grid, fine_η̄H, Center(), Center())
        end
        η̄H[I, J, k] = Σ / Azᶜᶜᶜ(I, J, k, grid)
        βᵘ[I, J, k] = restrict(I, J, k, Face(), Center(), cx, cy, grid, fine_grid, area_weighted, fine_βᵘ, Face(), Center()) / Azᶠᶜᶜ(I, J, k, grid)
        βᵛ[I, J, k] = restrict(I, J, k, Center(), Face(), cx, cy, grid, fine_grid, area_weighted, fine_βᵛ, Center(), Face()) / Azᶜᶠᶜ(I, J, k, grid)
    end
end

@kernel function _restrict!(bu, bv, ru, rv, η̄H, fine_η̄H, cx, cy, grid, fine_grid)
    I, J, k = @index(Global, NTuple)
    @inbounds begin
        bu[I, J, k] = ifelse(inactiveᶠᶜᶜ(I, J, k, grid, η̄H), zero(grid), restrict(I, J, k, Face(), Center(), cx, cy, grid, fine_grid, interior_u, ru, fine_η̄H))
        bv[I, J, k] = ifelse(inactiveᶜᶠᶜ(I, J, k, grid, η̄H), zero(grid), restrict(I, J, k, Center(), Face(), cx, cy, grid, fine_grid, interior_v, rv, fine_η̄H))
    end
end

@kernel function _prolongate!(xu, xv, cu, cv, η̄H, coarse_η̄H, cx, cy, grid, coarse_grid)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        xu[i, j, k] += ifelse(inactiveᶠᶜᶜ(i, j, k, grid, η̄H), zero(grid), prolongate(i, j, k, Face(), Center(), cx, cy, grid, coarse_grid, interior_u, cu, coarse_η̄H))
        xv[i, j, k] += ifelse(inactiveᶜᶠᶜ(i, j, k, grid, η̄H), zero(grid), prolongate(i, j, k, Center(), Face(), cx, cy, grid, coarse_grid, interior_v, cv, coarse_η̄H))
    end
end
