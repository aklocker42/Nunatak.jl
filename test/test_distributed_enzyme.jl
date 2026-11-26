using MPI
MPI.Init()

using Test
using Enzyme
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: ranks
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Nunatak
using Nunatak: ssa_operator!

include("toy_problems.jl")

# Run with `mpiexec -n 4`. Reverse-mode gradients on distributed grids must match the serial ones to round-off.

Nx, Ny = 32, 16

function coefficients(grid)
    η̄Hᶜᶜ = Field{Center, Center, Nothing}(grid)
    η̄Hᶠᶠ = Field{Face, Face, Nothing}(grid)
    βᵘ = Field{Face, Center, Nothing}(grid)
    βᵛ = Field{Center, Face, Nothing}(grid)
    η̄H(x, y) = 1 + x^2 + sin(2π * y)^2 / 2
    β(x, y) = 1 + x * y
    set!(η̄Hᶜᶜ, η̄H)
    set!(η̄Hᶠᶠ, η̄H)
    set!(βᵘ, β)
    set!(βᵛ, β)
    return η̄Hᶜᶜ, η̄Hᶠᶠ, βᵘ, βᵛ
end

function local_sum_of_squares(f)
    J = zero(eltype(f))
    for j in 1:size(f, 2), i in 1:size(f, 1)
        @inbounds J += f[i, j, 1]^2
    end
    return J
end

# Differences across the east and north halos make the cost depend on the neighbors' interiors.
function halo_cost(c)
    fill_halo_regions!(c)
    J = zero(eltype(c))
    for j in 1:size(c, 2), i in 1:size(c, 1)
        @inbounds J += (c[i+1, j, 1] - c[i, j, 1])^2 + (c[i, j+1, 1] - c[i, j, 1])^2 + (c[i+1, j+1, 1] - c[i, j, 1])^2
    end
    return J
end

function velocity_cost(x, solver, b, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ)
    fill_halo_regions!((η̄H, η̄Hᶠᶠ, βᵘ, βᵛ))
    fill!(x, 0)
    solve!(x, solver, b, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, FreeSlipWalls())
    return local_sum_of_squares(x.u) + local_sum_of_squares(x.v)
end

function halo_gradient(arch)
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, 1), y = (0, 1 / 2), topology = (Periodic, Periodic, Flat))
    c = first(coefficients(grid))
    c̄ = Enzyme.make_zero(c)
    autodiff(set_runtime_activity(Reverse), halo_cost, Active, Duplicated(c, c̄))
    return (c̄,)
end

function velocity_gradient(arch, preconditioner)
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, 1), y = (0, 1 / 2), topology = (Bounded, Bounded, Flat))
    p = coefficients(grid)
    p̄ = map(Enzyme.make_zero, p)
    b = toy_right_hand_side(grid)
    b̄ = Enzyme.make_zero(b)
    x = VelocityVector(grid)
    solver = VelocityConjugateGradientSolver(ssa_operator!, grid; preconditioner = preconditioner(grid),
                                             iteration_control = Convergence(; reltol = 1e-13, maxiter = 10_000))
    autodiff(set_runtime_activity(Reverse), velocity_cost, Active, Duplicated(x, Enzyme.make_zero(x)), Const(solver),
             Duplicated(b, b̄), map(Duplicated, p, p̄)...)
    return (p̄..., b̄.u, b̄.v)
end

function relative_error(serial, distributed, arch)
    ri, rj, _ = arch.local_index
    error = maximum(zip(serial, distributed)) do (s, d)
        s, d = Array(interior(s, :, :, 1)), Array(interior(d, :, :, 1))
        I₀, J₀ = (ri - 1) * (Nx ÷ arch.ranks[1]), (rj - 1) * (Ny ÷ arch.ranks[2])
        local_error = maximum(abs, d .- s[I₀ .+ (1:size(d, 1)), J₀ .+ (1:size(d, 2))]) / max(maximum(abs, s), eps())
        return ifelse(isnan(local_error), Inf, local_error)
    end
    return MPI.Allreduce(error, max, MPI.COMM_WORLD)
end

partitions = (Partition(2, 2), Partition(4, 1))

@testset "Halo fill gradient matches serial: $(join(ranks(partition), "×")) ranks" for partition in partitions
    arch = Distributed(CPU(); partition)
    @test relative_error(halo_gradient(CPU()), halo_gradient(arch), arch) < 1e-12
end

preconditioners = (diagonal = grid -> StressBalanceDiagonalPreconditioner(),
                   multigrid = StressBalanceMultigridPreconditioner)

@testset "Velocity gradient matches serial: $name preconditioner, $(join(ranks(partition), "×")) ranks" for (name, preconditioner) in pairs(preconditioners),
                                                                                                        partition in partitions
    arch = Distributed(CPU(); partition)
    serial = velocity_gradient(CPU(), preconditioner)
    @test relative_error(serial, velocity_gradient(arch, preconditioner), arch) < 1e-10
end
