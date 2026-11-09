using Test
using LinearAlgebra: dot, norm
using Oceananigans
using Oceananigans.Architectures: on_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Nunatak
using Documenter

const arch = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU" ? GPU() : CPU()
@info "Testing on" arch

include("toy_problems.jl")

@testset "Nunatak" begin
    include("test_sigma_grid.jl")
    include("test_conjugate_gradient.jl")
    include("test_ssa_operator.jl")
    include("test_glen_viscosity.jl")
    include("test_friction.jl")
    include("test_ice_mask.jl")
    include("test_multigrid_preconditioner.jl")
    include("test_grounding.jl")
    include("test_calving_front.jl")
    include("test_front_advance.jl")
    include("test_diva.jl")
    include("test_shallow_ice.jl")
    include("test_thickness.jl")
    include("test_thermodynamics.jl")
    include("test_rate_factor.jl")
    include("test_model.jl")

    if arch isa CPU
        include("test_enzyme.jl")

        @testset "Doctests" begin
            doctest(Nunatak; manual = false)
        end
    end
end
