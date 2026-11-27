using MPI
MPI.Init()

using Test
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: NCCLDistributed, ranks
using Nunatak

# Run with `mpiexec -n 4`; TEST_ARCHITECTURE=GPU puts one GPU on each rank.
# Serial and distributed models step in lockstep and must agree to round-off.
# The ice crosses rank boundaries: a wavy front at 32–48 km and, on the grounded slope, a grounding line near 3 km.

child_architecture = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()

# Loading NCCL before this file also tests NCCLDistributed.
distributed_architectures = isnothing(Base.get_extension(Oceananigans, :OceananigansNCCLExt)) ? (Distributed,) : (Distributed, NCCLDistributed)

Nx, Ny = 32, 16
L = 64e3
H₀ = 500.0
Nsteps = 3
Δt = 5e5
x_front(y) = 40e3 + 8e3 * sin(2π * y / (Ny * L / Nx))
grounded_bed(x, y) = 100 - 3e-2 * x
floating_bed(x, y) = -2000.0

function build_sheet(arch; bed, masking = grid -> PartialCellFront(grid), preconditioner = grid -> StressBalanceDiagonalPreconditioner())
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, L), y = (0, Ny * L / Nx),
                           topology = (Bounded, Bounded, Flat))
    velocity_solver = StressBalanceSolver(grid; preconditioner = preconditioner(grid),
                                          linear_iteration_control = Convergence(; reltol = 1e-10, maxiter = 10_000),
                                          picard_iteration_control = Convergence(; reltol = 1e-8, maxiter = 200))
    model = IceSheetModel(grid; velocity_solver, rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                          friction = WeertmanFriction(grid; coefficient = 1e6),
                          grounding = SubgridGroundedFraction(), bed, masking = masking(grid))
    set!(model; H = (x, y) -> ifelse(x < x_front(y), H₀, 0), u = (x, y) -> 1e-9 * x)
    return model
end

cpu(f) = Array(interior(f, :, :, 1))
snapshot(model) = (H = cpu(model.geometry.H), φ = cpu(model.geometry.φ),
                   u = cpu(model.velocities.u), v = cpu(model.velocities.v))

function relative_error(serial, distributed, arch)
    ri, rj, _ = arch.local_index
    nx, ny = size(distributed.H)
    I₀, J₀ = (ri - 1) * nx, (rj - 1) * ny
    error = maximum(keys(serial)) do key
        d = getproperty(distributed, key)
        s = getproperty(serial, key)
        # a rank owns size(d) faces starting at its first cell
        local_serial = s[I₀ .+ (1:size(d, 1)), J₀ .+ (1:size(d, 2))]
        local_error = maximum(abs, d .- local_serial) / max(maximum(abs, s), eps())
        return ifelse(isnan(local_error), Inf, local_error)
    end
    return MPI.Allreduce(error, max, MPI.COMM_WORLD)
end

calving_front(grid) = PartialCellFront(grid; calving = VonMisesCalving(grid; stress_threshold = 1e5))

multigrid(grid) = StressBalanceMultigridPreconditioner(grid; smoother = LineJacobi())
partly_gathered_multigrid(grid) = StressBalanceMultigridPreconditioner(grid; smoother = LineJacobi(), agglomeration_size = 128)

cases = ((name = "grounded slope", serial = (; bed = grounded_bed), distributed = (; bed = grounded_bed)),
         (name = "von Mises calving", serial = (; bed = grounded_bed, masking = calving_front),
                                      distributed = (; bed = grounded_bed, masking = calving_front)),
         (name = "multigrid", serial = (; bed = floating_bed, preconditioner = multigrid),
                              distributed = (; bed = floating_bed, preconditioner = multigrid)),
         (name = "multigrid gathered below 128 cells", serial = (; bed = floating_bed, preconditioner = partly_gathered_multigrid),
                                                       distributed = (; bed = floating_bed, preconditioner = partly_gathered_multigrid)))

@testset "$distributed matches serial on $child_architecture: $(case.name), $(join(ranks(partition), "×")) ranks" for distributed in distributed_architectures,
                                                                                                      partition in (Partition(2, 2), Partition(4, 1)), case in cases
    arch = distributed(child_architecture; partition)
    serial_model = build_sheet(child_architecture; case.serial...)
    distributed_model = build_sheet(arch; case.distributed...)
    for step in 0:Nsteps
        if step > 0
            time_step!(serial_model, Δt)
            time_step!(distributed_model, Δt)
        end
        @test relative_error(snapshot(serial_model), snapshot(distributed_model), arch) < 1e-10
    end
end

# Below the last level a rank can halve as the global grid would be, or once small, the levels are gathered and coarsened on one grid.
global_sizes(levels::Tuple{}) = ()
global_sizes(levels::Tuple{<:Nunatak.Agglomeration}) = global_sizes(first(levels).levels)
global_sizes(levels::Tuple) = (size(reconstruct_global_grid(first(levels).grid)), global_sizes(Base.tail(levels))...)

@testset "$distributed multigrid levels match serial: $(join(ranks(partition), "×")) ranks, gathered below $agglomeration_size cells" for distributed in distributed_architectures,
                                                                                    partition in (Partition(2, 2), Partition(4, 1)),
                                                                                    agglomeration_size in (0, 128, 2^16)
    grid(arch) = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, L), y = (0, Ny * L / Nx), topology = (Bounded, Bounded, Flat))
    serial = StressBalanceMultigridPreconditioner(grid(child_architecture); agglomeration_size)
    distributed_preconditioner = StressBalanceMultigridPreconditioner(grid(distributed(child_architecture; partition)); agglomeration_size)
    @test global_sizes(distributed_preconditioner.levels) == global_sizes(serial.levels)
end
