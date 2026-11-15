# MITgcm STREAMICE verification experiment halfpipe_streamice: a mostly floating shelf in a half-pipe channel, fed
# through the west edge, with no-slip side walls and a free calving front advancing for 2 years. Compared with
# halfpipe_streamice_reference.csv, from STREAMICE run with converged Picard iterations (streamice_max_nl_iter = 500).
# STREAMICE's west flux boundary is a no-slip face plus a thickness flux into the edge cell; here a no-slip wall
# plus the same flux as a mass source in the first column.
using Oceananigans, Oceananigans.Units, Nunatak, Printf, DelimitedFiles

year = years = 365days
ρᵢ, ρw, g = 917.0, 1028.0, 9.81
B, C, n = 700.0, 5.0, 3
inflow = 1.5e6 / year
Δx = 2kilometers
Nx, Ny = 39, 19

# Bed elevation across the channel at cell centers, from the experiment's topog.box (uniform along the channel).
bed_profile = [-624.774, -673.646, -720.509, -764.084, -803.184, -836.742, -863.842, -883.745, -895.908, -900.000,
               -895.908, -883.745, -863.842, -836.742, -803.184, -764.084, -720.509, -673.646, -624.774]

# STREAMICE's analytic 'PARAM' shelf: 300 m thick beyond 50 km, thickening quadratically upstream, front at 70 km.
initial_thickness(x, y) = ifelse(x < 70kilometers, 300 + 1000 * min(1, (max(50kilometers - x, 0) / 62kilometers)^2), zero(x))

function halfpipe_streamice(; arch = CPU(), Nσ = 10, Δt = 0.2years, stop_time = 2years, output = "halfpipe_streamice")
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, Nx * Δx), y = (0, Ny * Δx),
                           topology = (Bounded, Bounded, Flat))

    # STREAMICE: η = B²/2 ε̇^((1-n)/n) and τᵇ = C² u with B in Pa a^(1/n), C² in Pa a m⁻¹, ε̇ in a⁻¹.
    rheology = GlenFlowLaw(grid; rate_factor = (B^2)^(-n) / year, exponent = n, strain_rate_regularization = 1e-12 / year)
    friction = WeertmanFriction(grid; coefficient = C^2 * year, exponent = 1, velocity_regularization = 1e-12 / year)
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-10, maxiter = 10_000),
                                          picard_iteration_control = Convergence(; reltol = 1e-8, maxiter = 500))
    model = IceSheetModel(grid; Nσ, rheology, friction, velocity_solver,
                          bed = repeat(bed_profile', Nx),
                          stress_balance = DepthIntegrated(shear = DIVAShear(), side_walls = NoSlipWalls()),
                          grounding = SubgridGroundedFraction(), masking = PartialCellFront(grid),
                          constants = IceSheetConstants(grid; ice_density = ρᵢ, ocean_density = ρw, gravitational_acceleration = g),
                          surface_mass_balance = (x, y) -> ifelse(x < Δx, inflow / Δx, zero(x)))
    set!(model, H = initial_thickness)

    simulation = Simulation(model; Δt, stop_time)
    rows = NTuple{5, Float64}[]

    function diagnostics()
        H = Array(interior(model.geometry.H, :, :, 1))
        Hʳ = Array(interior(model.geometry.front_thickness, :, :, 1))
        μ = Array(interior(model.geometry.ice_mask, :, :, 1))
        u = Array(interior(model.velocities.u, :, :, 1))
        covered = @. ifelse(μ == 1, 1.0, ifelse(Hʳ > 0, H / max(Hʳ, 1e-12), 0.0))
        volume = sum(H) * Δx^2
        area = sum(covered) * Δx^2
        front = sum(covered[:, (Ny + 1) ÷ 2]) * Δx
        return (; H, u, volume, area, front, max_speed = maximum(u))
    end

    function progress(sim)
        d = diagnostics()
        push!(rows, (time(sim) / year, d.volume / 1e9, d.area / 1e6, d.front / kilometer, d.max_speed * year))
        @info @sprintf("t = %4.2f a, V = %8.2f km³, A = %7.1f km², centerline front = %6.2f km, max u = %7.1f m/a", rows[end]...)
        return nothing
    end

    add_callback!(simulation, progress, IterationInterval(1))
    run!(simulation)

    d = diagnostics()
    writedlm(output * "_timeseries.csv", [["t_years" "volume_km3" "area_km2" "centerline_front_km" "max_u_m_per_a"];
                                          reduce(vcat, [collect(r)' for r in rows])], ',')
    writedlm(output * "_thickness.csv", d.H, ',')
    writedlm(output * "_u.csv", d.u .* year, ',')
    return rows
end
