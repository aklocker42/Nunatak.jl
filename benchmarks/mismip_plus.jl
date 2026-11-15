# MISMIP+ (Asay-Davis et al., 2016, Geosci. Model Dev. 9, 2471–2497): a marine ice stream in a trough over a
# retrograde bed, spun up to steady state (Ice0) and then perturbed by sub-shelf melt that depends on the ice draft
# (Ice1r, 100 years), or by 100 m/a melt beyond x = 480 km (Ice2r, 100 years), each followed by recovery without
# melt (Ice1ra, Ice2ra). Compared with mismip_plus_streamice_reference.csv, the MITgcm STREAMICE submission to
# Cornford et al. (2020, The Cryosphere 14, 2283–2301), whose rate factor A and Schoof sliding law are used here.
# Ice is removed beyond the calving front at x = 640 km by a large melt rate in one extra column of floating ice.
using Oceananigans, Oceananigans.Units, Nunatak, Printf, DelimitedFiles
using Oceananigans.BoundaryConditions: fill_halo_regions!

year = years = 31556926seconds
ρᵢ, ρw, g = 918.0, 1028.0, 9.81
Lx, Ly = 640kilometers, 80kilometers
A = 2.0392e-17 / year
β² = 1e4 * year^(1 / 3)
α² = 0.5
accumulation = 0.3 / year
calving_rate = 1e5 / year

B₀, B₂, B₄, B₆, x̄ = -150.0, -728.8, 343.91, -50.57, 300kilometers
fᶜ, dᶜ, wᶜ, z_deep = 4kilometers, 500.0, 24kilometers, -720.0

along_flow_bed(x) = B₀ + B₂ * (x / x̄)^2 + B₄ * (x / x̄)^4 + B₆ * (x / x̄)^6
across_flow_bed(y) = dᶜ / (1 + exp(-2 * (y - Ly / 2 - wᶜ) / fᶜ)) + dᶜ / (1 + exp(2 * (y - Ly / 2 + wᶜ) / fᶜ))
mismip_plus_bed(x, y) = max(along_flow_bed(x) + across_flow_bed(y), z_deep)

# Melt rates as functions of thickness, bed and x; the model applies them to the floating fraction of each cell.
Ω, H₀ᶜ, z₀ = 0.2 / year, 75.0, -100.0
calving(x) = ifelse(x > Lx, calving_rate, zero(x))
no_melt(H, b, x) = calving(x)

function draft_melt(H, b, x)
    zᵈ = max(b, -ρᵢ / ρw * H)
    return calving(x) + Ω * tanh((zᵈ - b) / H₀ᶜ) * max(z₀ - zᵈ, zero(zᵈ))
end

calving_event_melt(H, b, x) = calving(x) + ifelse(x > 480kilometers, 100 / year, zero(x))

function mismip_plus_model(; arch = CPU(), Δx = 4kilometers, Nσ = 10)
    Nx, Ny = round(Int, Lx / Δx) + 1, round(Int, Ly / Δx)
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, Nx * Δx), y = (0, Ly),
                           topology = (Bounded, Bounded, Flat))
    rheology = GlenFlowLaw(grid; rate_factor = A, strain_rate_regularization = 1e-5 / year)
    friction = RegularizedCoulombFriction(grid; coefficient = β², coulomb_coefficient = α², exponent = 1 / 3,
                                          velocity_regularization = 1e-5 / year)
    # Two Picard iterations per step, lagged from the previous step's velocity, which changes little in a step.
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-8, maxiter = 20_000),
                                          picard_iteration_control = FixedIterations(2))
    model = IceSheetModel(grid; Nσ, rheology, friction, velocity_solver,
                          bed = mismip_plus_bed,
                          stress_balance = DepthIntegrated(shear = DIVAShear(), side_walls = FreeSlipWalls()),
                          grounding = SubgridGroundedFraction(),
                          constants = IceSheetConstants(grid; ice_density = ρᵢ, ocean_density = ρw, gravitational_acceleration = g),
                          surface_mass_balance = accumulation)
    return model
end

function update_melt!(sim, melt)
    model = sim.model
    m, H, b = model.basal_melt_rate, model.geometry.H, model.bed
    x = reshape(xnodes(model.grid, Center()), :, 1)
    interior(m, :, :, 1) .= melt.(interior(H, :, :, 1), interior(b, :, :, 1), x)
    fill_halo_regions!(m)
    return nothing
end

# Volume, volume above flotation and grounded area inside x < 640 km, and the grounding line on the two rows
# beside the centerline, where the height above flotation changes sign, interpolated linearly.
function diagnostics(model)
    grid = model.grid
    Δx = grid.Δxᶜᵃᵃ
    Nx, Ny = size(grid, 1) - 1, size(grid, 2)
    H = Array(interior(model.geometry.H, 1:Nx, :, 1))
    b = Array(interior(model.bed, 1:Nx, :, 1))
    φ = Array(interior(model.geometry.φ, 1:Nx, :, 1))
    Hᶠ = @. max(-ρw / ρᵢ * b, 0)
    hᵃ = H .- Hᶠ
    x = Array(xnodes(grid, Center()))[1:Nx]

    function grounding_line(j)
        i = findlast(i -> hᵃ[i, j] > 0 && hᵃ[i+1, j] ≤ 0, 1:Nx-1)
        return isnothing(i) ? NaN : x[i] + Δx * hᵃ[i, j] / (hᵃ[i, j] - hᵃ[i+1, j])
    end

    volume = sum(H) * Δx^2
    volume_above_flotation = sum(max.(hᵃ, 0)) * Δx^2
    grounded_area = sum(φ) * Δx^2
    centerline_grounding_line = (grounding_line(Ny ÷ 2) + grounding_line(Ny ÷ 2 + 1)) / 2
    return (; H, volume, volume_above_flotation, grounded_area, centerline_grounding_line)
end

function record!(sim, (rows, experiment))
    d = diagnostics(sim.model)
    push!(rows, (experiment, time(sim) / year, d.volume, d.volume_above_flotation, d.grounded_area, d.centerline_grounding_line))
    @info @sprintf("%-7s t = %7.1f a, V = %.4e m³, VAF = %.4e m³, grounded = %.4e m², centerline GL = %6.1f km, wall time: %s",
                   rows[end][1:5]..., rows[end][6] / kilometer, prettytime(sim.run_wall_time))
    return nothing
end

function run_experiment!(rows, experiment, model, melt, stop_time; Δt, report_interval)
    simulation = Simulation(model; Δt, stop_time)
    add_callback!(simulation, update_melt!, IterationInterval(1); parameters = melt)
    add_callback!(simulation, record!, TimeInterval(report_interval); parameters = (rows, experiment))
    run!(simulation)
    return model
end

function restarted_model(H, time; kw...)
    model = mismip_plus_model(; kw...)
    model.clock.time = time
    set!(model, H = H)
    return model
end

# Spin up Ice0 from a uniform 100 m slab for `spinup_time`, or start from `initial_thickness` (an `Nx × Ny` matrix
# including the calving column), then run Ice0, Ice1r, Ice1ra, Ice2r and Ice2ra; STREAMICE's recovery runs end at 1000 a.
function mismip_plus(; arch = CPU(), Δx = 4kilometers, Nσ = 10, spinup_time = 20_000years, initial_thickness = nothing,
                       recovery_time = 1000years, Δt = min(0.5, Δx / 4kilometers) * year, output = "mismip_plus")
    kw = (; arch, Δx, Nσ)
    stepping = (; Δt)

    spinup = NTuple{6, Any}[]
    if isnothing(initial_thickness)
        model = mismip_plus_model(; kw...)
        set!(model, H = (x, y) -> ifelse(x < Lx, 100.0, 0.0))
        run_experiment!(spinup, "spinup", model, no_melt, spinup_time; stepping..., report_interval = 1000years)
        initial_thickness = diagnostics(model).H
        initial_thickness = vcat(initial_thickness, zeros(1, size(initial_thickness, 2)))
        writedlm(output * "_Ice0_thickness.csv", initial_thickness, ',')
    end

    rows = NTuple{6, Any}[]
    run_experiment!(rows, "Ice0", restarted_model(initial_thickness, 0.0; kw...), no_melt, 100years; stepping..., report_interval = 10years)

    for (retreat, recovery, melt) in (("Ice1r", "Ice1ra", draft_melt), ("Ice2r", "Ice2ra", calving_event_melt))
        model = restarted_model(initial_thickness, 0.0; kw...)
        run_experiment!(rows, retreat, model, melt, 100years; stepping..., report_interval = 10years)
        H = vcat(diagnostics(model).H, zeros(1, size(initial_thickness, 2)))
        model = restarted_model(H, model.clock.time; kw...)
        run_experiment!(rows, recovery, model, no_melt, recovery_time; stepping...,
                        report_interval = 10years)
    end

    header = ["experiment" "t_years" "volume_m3" "volume_above_flotation_m3" "grounded_area_m2" "centerline_grounding_line_m"]
    writedlm(output * "_timeseries.csv", [header; reduce(vcat, [permutedims(collect(r)) for r in rows])], ',')

    reference = readdlm(joinpath(@__DIR__, "mismip_plus_streamice_reference.csv"), ','; header = true)[1]
    println("\nCenterline grounding line (km) and volume above flotation (10¹³ m³), Nunatak vs STREAMICE")
    for r in rows
        k = findfirst(i -> reference[i, 1] == r[1] && isapprox(reference[i, 2], r[2]; atol = 0.5), axes(reference, 1))
        isnothing(k) && continue
        @printf("%-7s t = %6.0f a  GL %6.1f %6.1f  VAF %.4f %.4f\n", r[1], r[2], r[6] / kilometer, reference[k, 6] / kilometer,
                r[4] / 1e13, reference[k, 4] / 1e13)
    end
    return rows
end
