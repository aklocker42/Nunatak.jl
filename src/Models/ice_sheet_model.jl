"""
$(TYPEDEF)

Depth-integrated (SSA/DIVA) ice-sheet model on a horizontal Oceananigans grid with a
fixed σ column for the vertical velocity structure.

$(TYPEDFIELDS)
"""
struct IceSheetModel{TS, Arch, G, SG, C, SB, R, F, GR, TH, U, GE, B, A, M, AU, VS} <: AbstractModel{TS, Arch}
    "architecture"
    architecture :: Arch
    "horizontal grid (Flat in z)"
    grid :: G
    "σ column grid"
    σgrid :: SG
    "clock"
    clock :: C
    "`DepthIntegrated` stress balance"
    stress_balance :: SB
    "rheology, e.g. `GlenFlowLaw`"
    rheology :: R
    "basal friction law"
    friction :: F
    "grounding-line treatment"
    grounding :: GR
    "thermodynamics (reserved, `nothing` in phase 0)"
    thermodynamics :: TH
    "velocities `(u, v, uᵇ, vᵇ, uσ, vσ)`"
    velocities :: U
    "geometry `(H, s, φ, φᵘ, φᵛ, N, ice_mask)`"
    geometry :: GE
    "bed elevation"
    bed :: B
    "surface mass balance (m s⁻¹ ice equivalent)"
    surface_mass_balance :: A
    "basal melt rate (m s⁻¹ ice equivalent, positive melts)"
    basal_melt_rate :: M
    "stress-balance diagnostics"
    auxiliary :: AU
    "velocity solver"
    velocity_solver :: VS
    "time stepper"
    timestepper :: TS
end

"""
$(TYPEDSIGNATURES)

Build an `IceSheetModel` on the horizontal `grid` (z must be `Flat`).

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (8, 8), x = (0, 1e5), y = (0, 1e5), topology = (Bounded, Bounded, Flat));

julia> model = IceSheetModel(grid; rheology = GlenFlowLaw(grid; rate_factor = 1e-24));

julia> summary(model)
"IceSheetModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)"
```
"""
function IceSheetModel(grid;
                       Nσ = 10,
                       bed_refinement = 2,
                       clock = Clock(grid),
                       stress_balance = DepthIntegrated(),
                       rheology = nothing,
                       friction = nothing,
                       grounding = nothing,
                       thermodynamics = nothing,
                       bed = 0,
                       surface_mass_balance = 0,
                       basal_melt_rate = 0,
                       velocity_solver = nothing,
                       timestepper = :SplitRungeKutta3)

    σgrid = SigmaGrid(grid; Nσ, bed_refinement)
    velocities = ice_sheet_velocities(grid, σgrid)
    geometry = ice_sheet_geometry(grid)
    auxiliary = ice_sheet_auxiliary_fields(grid, σgrid)

    bed = parameter_field(grid, bed)
    surface_mass_balance = parameter_field(grid, surface_mass_balance)
    basal_melt_rate = parameter_field(grid, basal_melt_rate)

    timestepper = TimeStepper(timestepper, grid, (; H = geometry.H))

    return IceSheetModel(architecture(grid), grid, σgrid, clock, stress_balance, rheology, friction,
                         grounding, thermodynamics, velocities, geometry, bed, surface_mass_balance,
                         basal_melt_rate, auxiliary, velocity_solver, timestepper)
end
