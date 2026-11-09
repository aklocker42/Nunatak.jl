module Nunatak

export SigmaGrid, sigma_faces
export IceSheetModel
export DepthIntegrated, DIVAShear, FreeSlipWalls, NoSlipWalls, ShallowIce, ShallowIceSolver
export GlenFlowLaw, ArrheniusRateFactor
export EnthalpyThermodynamics
export WeertmanFriction, RegularizedCoulombFriction
export SubgridGroundedFraction, IceSheetConstants, SmoothIceMask, PartialCellFront
export StressBalanceSolver
export VelocityVector, VelocityConjugateGradientSolver, JacobiPreconditioner, StressBalanceDiagonalPreconditioner,
       StressBalanceMultigridPreconditioner
export Convergence, FixedIterations, PicardIteration
export solve!, adjoint_solve!, precondition!, picard_solve!
export checkpointed_gradient!

using Adapt
using DocStringExtensions
using KernelAbstractions: @kernel, @index
using LinearAlgebra: LinearAlgebra

using Oceananigans
using Oceananigans: AbstractModel
using Oceananigans.Advection: div_Uc, UpwindBiased, BoundsPreservingWENO, materialize_advection, update_bounds_preserving_limiter!,
                             _advective_tracer_flux_x, _advective_tracer_flux_y, _biased_interpolate_xᶠᵃᵃ, _biased_interpolate_yᵃᶠᵃ,
                             LeftBias, RightBias, upwind_biased_product, rescaled_reconstruction
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: interior, ZeroField
using Oceananigans.Grids: AbstractGrid, RectilinearGrid, Flat, Bounded, Periodic, topology, xnode, ynode, halo_size, znode,
                          cpu_face_constructor_x, cpu_face_constructor_y, pop_flat_elements, peripheral_node, inactive_cell
using Oceananigans.Solvers: BatchedTridiagonalSolver
using Oceananigans.Operators: δxᶠᶜᶜ, δyᶠᶜᶜ, δxᶜᶠᶜ, δyᶜᶠᶜ, ∂xᶜᶜᶜ, ∂yᶜᶜᶜ, ∂xᶠᶠᶜ, ∂yᶠᶠᶜ, ∂xᶠᶜᶜ, ∂yᶜᶠᶜ,
                              Δxᶠᶠᶜ, Δyᶠᶠᶜ, Δxᶠᶜᶜ, Δyᶠᶜᶜ, Δxᶜᶠᶜ, Δyᶜᶠᶜ, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Az, Azᶠᶜᶜ, Azᶜᶠᶜ, Azᶜᶜᶜ, Axᶠᶜᶜ, Ayᶜᶠᶜ, Vᶜᶜᶜ, δxᶜᵃᵃ, δyᵃᶜᵃ,
                              ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ℑxyᶜᶜᵃ, ℑxyᶠᶠᵃ, ℑxyᶠᶜᵃ, ℑxyᶜᶠᵃ
using Oceananigans.TimeSteppers: TimeStepper, Clock, SplitRungeKuttaTimeStepper
using Oceananigans.Utils: launch!, prettysummary

import Oceananigans.Advection: cell_advection_timescale
import Oceananigans.Diagnostics: cell_diffusion_timescale
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
include("StressBalances/strain_rates.jl")
include("StressBalances/ssa_operator.jl")
include("StressBalances/driving_stress.jl")
include("Rheologies/arrhenius_rate_factor.jl")
include("Rheologies/glen_flow_law.jl")
include("Frictions/frictions.jl")
include("StressBalances/stress_balance_coefficients.jl")
include("StressBalances/diva.jl")
include("Models/ice_sheet_constants.jl")
include("Groundings/subgrid_grounded_fraction.jl")
include("Solvers/iteration_controls.jl")
include("Solvers/velocity_vector.jl")
include("Solvers/conjugate_gradient.jl")
include("StressBalances/diagonal_preconditioner.jl")
include("StressBalances/multigrid_preconditioner.jl")
include("Solvers/picard.jl")
include("StressBalances/stress_balance_solver.jl")
include("StressBalances/shallow_ice.jl")
include("Models/smooth_ice_mask.jl")
include("Models/partial_cell_front.jl")
include("Thermodynamics/enthalpy_thermodynamics.jl")
include("Thermodynamics/enthalpy_conversions.jl")
include("Thermodynamics/layer_transport.jl")
include("Thermodynamics/strain_heating.jl")
include("Thermodynamics/column_energy_solve.jl")
include("Models/ice_sheet_fields.jl")
include("Models/checkpointed_gradient.jl")
include("Models/ice_sheet_model.jl")
include("Models/ice_sheet_model_methods.jl")
include("Models/time_stepping.jl")

end # module Nunatak
