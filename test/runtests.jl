using Test
using LinearAlgebra: dot, norm
using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Nunatak
using Documenter

include("toy_problems.jl")

@testset "Nunatak" begin
    include("test_sigma_grid.jl")
    include("test_conjugate_gradient.jl")
    include("test_model.jl")
    include("test_enzyme.jl")

    @testset "Doctests" begin
        doctest(Nunatak; manual = false)
    end
end
