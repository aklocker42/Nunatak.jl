# Steady marine flowline on a prograde bed against the boundary-layer grounding-line position of Schoof (2007,
# J. Geophys. Res. 112), Weertman friction, no buttressing. The grounding line is approached from upstream (`start = :advance`,
# thin ice everywhere) and from downstream (`start = :retreat`, thick ice grounded past the theoretical position);
# the gap between the two steady positions measures the resolution error. `band` refines Δx between x₁ and x₂.
using Oceananigans, Oceananigans.Units, Nunatak, Printf, DelimitedFiles
using Oceananigans.Operators: Δxᶜᶜᶜ
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: xnode

year = years = 31557600seconds

schoof_bed(x, y) = -100 - 6e-4 * x

function schoof_grounding_line(A, C, a; ρᵢ = 918.0, ρw = 1028.0, g = 9.81, n = 3, m = 1 / 3)
    q(h) = (A * (ρᵢ * g)^(n + 1) * (1 - ρᵢ / ρw)^n / (4^n * C))^(1 / (m + 1)) * h^((m + n + 3) / (m + 1))
    h(x) = -ρw / ρᵢ * schoof_bed(x, 0)
    x₁, x₂ = 1e3, 5e6
    for _ in 1:100
        x = (x₁ + x₂) / 2
        a * x > q(h(x)) ? (x₁ = x) : (x₂ = x)
    end
    return (x₁ + x₂) / 2
end

# Faces with spacing Δx between x₁ and x₂, Δxᶜ elsewhere, joined by tanh transitions of width w.
function band_faces(Δx, x_end; Δxᶜ = 8kilometers, x₁ = 280kilometers, x₂ = 560kilometers, w = 30kilometers)
    spacing(x) = Δxᶜ - (Δxᶜ - Δx) * (tanh((x - x₁) / w) - tanh((x - x₂) / w)) / 2
    faces = [0.0]
    while faces[end] < x_end
        push!(faces, faces[end] + spacing(faces[end]))
    end
    return faces
end

function schoof_flowline(; arch = CPU(), Δx = 8kilometers, band = false, driving_stress = false, start = :advance,
                           A = 2e-17, C = 1e4, a = 0.3, stop_time = 50_000years, report_interval = 1000years,
                           output = "schoof_flowline")
    ρᵢ, ρw = 918.0, 1028.0
    x_front = 800kilometers
    x_GL = schoof_grounding_line(A, C, a; ρᵢ, ρw)
    Δxᶜ = band ? 8kilometers : Δx
    x = band ? band_faces(Δx, x_front + 8Δxᶜ; Δxᶜ) : (0, round(Int, (x_front + 8Δx) / Δx) * Δx)
    Nx = band ? length(x) - 1 : round(Int, (x_front + 8Δx) / Δx)
    Ny = 4
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x, y = (0, Ny * Δx), topology = (Bounded, Bounded, Flat))

    rheology = GlenFlowLaw(grid; rate_factor = A / year, strain_rate_regularization = 1e-5 / year)
    friction = WeertmanFriction(grid; coefficient = C * year^(1 / 3), velocity_regularization = 1e-5 / year)
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-8, maxiter = 20_000),
                                          picard_iteration_control = FixedIterations(2))
    model = IceSheetModel(grid; velocity_solver, rheology, friction,
                          grounding = SubgridGroundedFraction(; driving_stress),
                          constants = IceSheetConstants(grid; ice_density = ρᵢ, ocean_density = ρw),
                          bed = schoof_bed, surface_mass_balance = a / year,
                          basal_melt_rate = (x, y) -> ifelse(x > x_front, 1, 0))

    x_start = 1.3 * x_GL
    retreat_thickness(x) = ifelse(x < x_start, -1.2 * ρw / ρᵢ * schoof_bed(x, 0) + 1500 * sqrt(max(1 - x / x_start, 0)), 200)
    set!(model; H = (x, y) -> ifelse(x < x_front, start == :retreat ? retreat_thickness(x) : 100, 0))

    simulation = Simulation(model; Δt = Δx / 4000 * year, stop_time)

    cpu_grid = on_architecture(CPU(), grid)
    widths = [Δxᶜᶜᶜ(i, 1, 1, cpu_grid) * (xnode(i, 1, 1, cpu_grid, Center(), Center(), Center()) < x_front) for i in 1:Nx]
    grounding_line_position() = sum(widths .* Array(interior(model.geometry.φ, :, 2, 1)))

    rows = NTuple{3, Float64}[]
    function progress(sim)
        t = time(sim) / year
        x = grounding_line_position()
        drift = isempty(rows) ? x : x - rows[end][2]
        push!(rows, (t, x, drift))
        @info @sprintf("t = %6.0f a, grounding line = %7.2f km (Schoof %7.2f km), drift = %+.3f km per %d a, wall time: %s",
                       t, x / 1e3, x_GL / 1e3, drift / 1e3, report_interval / year, prettytime(sim.run_wall_time))
        sim.running = !(t > 5000 && abs(drift) < 20)
        return nothing
    end

    add_callback!(simulation, progress, TimeInterval(report_interval))
    run!(simulation)

    writedlm(output * "_timeseries.csv", vcat(["t (a)" "grounding line (m)" "drift (m)"], reduce(vcat, [collect(r)' for r in rows])), ',')
    return (; model, grounding_line = rows[end][2], schoof = x_GL)
end
