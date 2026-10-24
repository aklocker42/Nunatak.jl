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
using Oceananigans: AbstractModel
using Oceananigans.Fields: interior
using Oceananigans.Grids: AbstractGrid, RectilinearGrid, Flat, Bounded, topology, halo_size, znode,
                          cpu_face_constructor_x, cpu_face_constructor_y, pop_flat_elements, peripheral_node
using Oceananigans.Operators: δxᶠᶜᶜ, δyᶠᶜᶜ, δxᶜᶠᶜ, δyᶜᶠᶜ, ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂xᶠᶠᶜ, ∂yᶠᶠᶜ,
                              Δxᶠᶠᶜ, Δyᶠᶠᶜ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Azᶠᶜᶜ, Azᶜᶠᶜ
using Oceananigans.TimeSteppers: TimeStepper, Clock, SplitRungeKuttaTimeStepper
using Oceananigans.Utils: launch!, prettysummary

import Oceananigans: fields, prognostic_fields, initialize!, prognostic_state, restore_prognostic_state!
import Oceananigans.Architectures: architecture
import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.Fields: set!
import Oceananigans.OutputWriters: checkpointer_address, default_included_properties
import Oceananigans.Simulations: iteration
import Oceananigans.TimeSteppers: update_state!, reset!, cache_current_fields!, rk_substep!
import Oceananigans.Utils: prettytime

include("Grids/sigma_grid.jl")
include("Fields/parameter_fields.jl")
include("StressBalances/depth_integrated.jl")
include("StressBalances/column_integrals.jl")
include("StressBalances/ssa_operator.jl")
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
