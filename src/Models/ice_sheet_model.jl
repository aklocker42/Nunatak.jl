"""
$(TYPEDEF)

Depth-integrated (SSA/DIVA) ice-sheet model on a horizontal Oceananigans grid with a
fixed σ column for the vertical velocity structure.

$(TYPEDFIELDS)
"""
struct IceSheetModel{TS, Arch, G, SG, C, PC, SB, R, F, GR, MK, TH, U, GE, B, A, M, AU, VS, AD} <: AbstractModel{TS, Arch}
    "architecture"
    architecture :: Arch
    "horizontal grid (Flat in z)"
    grid :: G
    "σ column grid"
    σgrid :: SG
    "clock"
    clock :: C
    "`IceSheetConstants`"
    constants :: PC
    "`DepthIntegrated` stress balance"
    stress_balance :: SB
    "rheology, e.g. `GlenFlowLaw`"
    rheology :: R
    "basal friction law"
    friction :: F
    "grounding-line treatment"
    grounding :: GR
    "ice mask, e.g. `SmoothIceMask`"
    masking :: MK
    "`EnthalpyThermodynamics`, or `nothing`"
    thermodynamics :: TH
    "velocities `(u, v, uᵇ, vᵇ, uσ, vσ)`"
    velocities :: U
    "geometry `(H, s, φ, φᵘ, φᵛ, N, ice_mask, clipped_thickness)`"
    geometry :: GE
    "bed elevation"
    bed :: B
    "surface mass balance (m s⁻¹ ice equivalent)"
    surface_mass_balance :: A
    "basal melt rate (m s⁻¹ ice equivalent, positive melts)"
    basal_melt_rate :: M
    "stress-balance diagnostics"
    auxiliary :: AU
    "`StressBalanceSolver`, or `nothing` to keep the velocities fixed"
    velocity_solver :: VS
    "time stepper"
    timestepper :: TS
    "advection scheme for the ice thickness"
    advection :: AD
end

"""
$(TYPEDSIGNATURES)

Build an `IceSheetModel` on the horizontal `grid` (z must be `Flat`). With a `rheology` the
velocities are solved by a `StressBalanceSolver` in `update_state!`; pass `velocity_solver = nothing`
to keep them fixed. The thickness is transported by the depth-averaged velocity with `advection`,
fifth-order WENO limited to `0 ≤ H ≤ 100 km` by default, which needs a horizontal halo of at least 4;
`UpwindBiased(order = 1)` works with any halo. Cells thinner than `masking.minimum_thickness` are
ice free and drop out of the stress balance; see `SmoothIceMask`. `masking = PartialCellFront(grid)` instead
advances calving fronts with partially filled cells.

With `stress_balance = ShallowIce()` the velocities follow from the local surface slope through a
`ShallowIceSolver` (no linear solve), the default thickness advection is `Centered(order = 2)` so the
flux is the diffusive flux of Mahaffy (1976), and `friction` must be `nothing` or a `WeertmanFriction`.
`cell_diffusion_timescale(model)` then bounds the stable explicit time step.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (8, 8), halo = (4, 4), x = (0, 1e5), y = (0, 1e5), topology = (Bounded, Bounded, Flat));

julia> model = IceSheetModel(grid; rheology = GlenFlowLaw(grid; rate_factor = 1e-24));

julia> summary(model)
"IceSheetModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)"
```
"""
function IceSheetModel(grid;
                       Nσ = 10,
                       bed_refinement = 2,
                       clock = Clock(grid),
                       constants = IceSheetConstants(grid),
                       stress_balance = DepthIntegrated(),
                       rheology = nothing,
                       friction = nothing,
                       grounding = nothing,
                       masking = SmoothIceMask(grid),
                       thermodynamics = nothing,
                       bed = 0,
                       surface_mass_balance = 0,
                       basal_melt_rate = 0,
                       velocity_solver = default_velocity_solver(grid, rheology, stress_balance),
                       advection = default_thickness_advection(grid, stress_balance),
                       timestepper = :SplitRungeKutta3)

    validate_stress_balance(stress_balance, rheology, friction)
    validate_velocity_solver(stress_balance, velocity_solver)
    σgrid = SigmaGrid(grid; Nσ, bed_refinement)
    velocities = ice_sheet_velocities(grid, σgrid)
    geometry = ice_sheet_geometry(grid)
    auxiliary = ice_sheet_auxiliary_fields(grid, σgrid)

    bed = parameter_field(grid, bed)
    surface_mass_balance = parameter_field(grid, surface_mass_balance)
    basal_melt_rate = parameter_field(grid, basal_melt_rate)
    thermodynamics = materialize_thermodynamics(thermodynamics, grid, σgrid, clock)
    rheology = materialize_rheology(rheology, σgrid, thermodynamics)

    timestepper = TimeStepper(timestepper, grid, (; H = geometry.H))
    check_advection_halo(advection, grid)
    advection = materialize_advection(advection, grid)

    model = IceSheetModel(architecture(grid), grid, σgrid, clock, constants, stress_balance, rheology, friction,
                          grounding, masking, thermodynamics, velocities, geometry, bed, surface_mass_balance,
                          basal_melt_rate, auxiliary, velocity_solver, timestepper, advection)

    update_thermal_diagnostics!(model, thermodynamics)
    return model
end

# The finite upper bound keeps the limiter's reverse-mode derivative finite; Inf gives 0 * Inf = NaN.
default_thickness_advection(grid, stress_balance) = WENO(eltype(grid); order = 5, bounds = (zero(grid), convert(eltype(grid), 1e5)))

# Centered face thickness makes the shallow-ice flux the diffusive flux of Mahaffy (1976).
default_thickness_advection(grid, ::ShallowIce) = Centered(eltype(grid); order = 2)

check_advection_halo(advection, grid) = nothing

# The limiter is also computed in the first halo layer, so its stencil reaches one cell further.
function check_advection_halo(advection::BoundsPreservingWENO{N}, grid) where N
    Hx, Hy, _ = halo_size(grid)
    min(Hx, Hy) > N || throw(ArgumentError("bounded WENO(order = $(2N - 1)) needs a horizontal halo of at least $(N + 1), got ($Hx, $Hy)"))
    return nothing
end
