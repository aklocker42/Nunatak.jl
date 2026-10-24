module Nunatak

export SigmaGrid, sigma_faces
export IceSheetModel
export DepthIntegrated, DIVAShear
export GlenFlowLaw
export WeertmanFriction, RegularizedCoulombFriction
export SubgridGroundedFraction
export VelocityVector, VelocityConjugateGradientSolver, JacobiPreconditioner
export Convergence, FixedIterations, PicardIteration
export solve!, adjoint_solve!, precondition!, picard_solve!

using Adapt
using DocStringExtensions
using KernelAbstractions: @kernel, @index
using LinearAlgebra: LinearAlgebra

using Oceananigans
using Oceananigans: AbstractModel, prognostic_state, restore_prognostic_state!
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: interior
using Oceananigans.Grids: RectilinearGrid, Flat, Bounded, topology, halo_size
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: TimeStepper, Clock

include("Grids/sigma_grid.jl")
include("Fields/parameter_fields.jl")
include("StressBalances/depth_integrated.jl")
include("StressBalances/column_integrals.jl")
include("Rheologies/glen_flow_law.jl")
include("Frictions/frictions.jl")
include("Groundings/subgrid_grounded_fraction.jl")
include("Solvers/iteration_controls.jl")
include("Solvers/velocity_vector.jl")
include("Solvers/conjugate_gradient.jl")
include("Solvers/picard.jl")
include("Models/ice_sheet_fields.jl")
include("Models/ice_sheet_model.jl")
include("Models/ice_sheet_model_methods.jl")
include("Models/time_stepping.jl")

end # module Nunatak
