# CalvingMIP circular (Exp1, Exp2) and Thule (Exp3, Exp4) experiments
# (https://github.com/JRowanJordan/CalvingMIP/wiki). Exp1 and Exp3 spin up for 10 000 a behind a front held at
# 750 km; Exp2 and Exp4 restart from their latest checkpoint and move the front at a prescribed velocity.
# `calvingmip_netcdf` writes the CalvingMIP results file, `plot_calvingmip` the validation figure, and
# `calvingmip_summary` the front displacement against the prescribed one.
using Oceananigans, Oceananigans.Units, Nunatak, Printf, JLD2, NCDatasets, CairoMakie

year = 31556926seconds
R = 800kilometers
front_radius = 750kilometers
ρᵢ, ρₒ = 917.0, 1028.0
accumulation_rate = 0.3

circular_bed(x, y) = 900 - 2900 * (x^2 + y^2) / R^2

function thule_bed(x, y)
    r, θ = sqrt(x^2 + y^2), atan(y, x)
    l = R - cos(2θ) * R / 2
    return circular_bed(x, y) + 1100 * cos(3π * r / l)
end

calvingmip_bed(experiment) = experiment in (1, 2) ? circular_bed : thule_bed
calvingmip_name(experiment, Δx) = @sprintf("calvingmip_exp%d_dx%gkm", experiment, Δx / kilometers)

function calvingmip_grid(arch, Δx)
    Nx = 2 * round(Int, R / Δx) + 1
    return RectilinearGrid(arch; size = (Nx, Nx), halo = (4, 4), x = (-Nx * Δx / 2, Nx * Δx / 2), y = (-Nx * Δx / 2, Nx * Δx / 2),
                           topology = (Bounded, Bounded, Flat))
end

held_front(grid) = MaximumExtentCalving(grid; extent = (x, y) -> front_radius - sqrt(x^2 + y^2))

function calvingmip_model(grid, bed, calving)
    rheology = GlenFlowLaw(grid; rate_factor = 9.31e-26, strain_rate_regularization = 1e-5 / year)
    friction = WeertmanFriction(grid; coefficient = (1e-3 / year / 1e9)^(-1 / 3), exponent = 1 / 3,
                                velocity_regularization = 1e-5 / year)
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-8, maxiter = 20_000),
                                          picard_iteration_control = FixedIterations(2))
    return IceSheetModel(grid; rheology, friction, velocity_solver, bed,
                         grounding = SubgridGroundedFraction(),
                         masking = PartialCellFront(grid; calving),
                         constants = IceSheetConstants(grid; ice_density = ρᵢ, ocean_density = ρₒ),
                         surface_mass_balance = accumulation_rate / year)
end

function latest_checkpoint(prefix)
    files = filter(f -> startswith(f, prefix * "_iteration") && endswith(f, ".jld2"), readdir("."))
    isempty(files) && throw(ArgumentError("no checkpoint $(prefix)_iteration*.jld2 here; run the experiment it restarts from first"))
    return argmax(f -> parse(Int, match(r"iteration(\d+)", f)[1]), files)
end

# Spin-ups write a snapshot every 100 a; moving-front runs every year, for the yearly CalvingMIP outputs.
function run_calvingmip!(model, prefix, stop_time; checkpoint = nothing, start = 0, spinup = false)
    Δx = model.grid.Δxᶠᵃᵃ
    simulation = Simulation(model; Δt = 1year, stop_time)
    if !isnothing(checkpoint)
        set!(simulation; checkpoint)
        time(simulation) ≈ start || error(@sprintf("%s is at %.1f a, not %.1f a", checkpoint, time(simulation) / year, start / year))
        @info @sprintf("Restarted from %s at %.1f a", checkpoint, time(simulation) / year)
    end

    function progress(sim)
        m = sim.model
        area = sum(Array(interior(m.geometry.ice_mask)) .> 0) * Δx^2
        volume = sum(Array(interior(m.geometry.H))) * Δx^2
        @info @sprintf("t = %7.1f a, front radius %6.1f km, volume %.4e km³, max |u| %7.1f m/a, wall %s",
                       time(sim) / year, sqrt(area / π) / kilometers, volume / kilometers^3,
                       maximum(abs, m.velocities.u) * year, prettytime(sim.run_wall_time))
        return nothing
    end

    add_callback!(simulation, progress, TimeInterval(50year))
    outputs = (; H = model.geometry.H, μ = model.geometry.ice_mask, Hʳ = model.geometry.front_thickness, φ = model.geometry.φ,
                 C = model.masking.calved_thickness, u = model.velocities.u, v = model.velocities.v)
    filename = prefix * "_fields.jld2"
    simulation.output_writers[:fields] = JLD2Writer(model, outputs; filename, overwrite_files = true,
                                                    schedule = TimeInterval(spinup ? 100year : 1year))
    simulation.output_writers[:checkpointer] = Checkpointer(model; prefix, cleanup = false,
                                                            schedule = TimeInterval(spinup ? 1000year : 500year))
    run!(simulation)
    return filename
end

# Exp4 moves the front for 500 a, then holds it inside 750 km for another 500 a as a second run from its checkpoint.
function calvingmip(experiment; arch = CPU(), Δx = 5kilometers, spinup_time = 10_000year)
    experiment in 1:4 || throw(ArgumentError("experiment must be 1, 2, 3 or 4, not $experiment"))
    grid = calvingmip_grid(arch, Δx)
    bed = calvingmip_bed(experiment)
    prefix = calvingmip_name(experiment, Δx)

    if experiment in (1, 3)
        model = calvingmip_model(grid, bed, held_front(grid))
        set!(model, H = (x, y) -> ifelse(x^2 + y^2 < front_radius^2, 100.0, 0.0))
        return [run_calvingmip!(model, prefix, spinup_time; spinup = true)]
    end

    t₀ = spinup_time
    Wᵐᵃˣ = experiment == 2 ? 300 : 750
    wᵥ(t) = -Wᵐᵃˣ / year * sin(2π * (t - t₀) / 1000year)
    model = calvingmip_model(grid, bed, FrontVelocityCalving(grid; front_velocity = wᵥ))
    duration = experiment == 2 ? 1000year : 500year
    files = [run_calvingmip!(model, prefix, t₀ + duration; checkpoint = latest_checkpoint(calvingmip_name(experiment - 1, Δx)), start = t₀)]
    if experiment == 4
        free = calvingmip_model(grid, bed, held_front(grid))
        push!(files, run_calvingmip!(free, prefix * "_free", t₀ + 1000year; checkpoint = latest_checkpoint(prefix), start = t₀ + 500year))
    end
    return files
end

# Profiles as (name, start, end, spacing): rays from the centre on the circular domain, the Caprona and Halbrane lines
# on Thule.
circular_profiles() = [(name, (0.0, 0.0), 800e3 .* direction, 5e3)
                       for (name, direction) in (("A", (0.0, 1.0)), ("B", (1.0, 1.0) ./ √2), ("C", (1.0, 0.0)), ("D", (1.0, -1.0) ./ √2),
                                                 ("E", (0.0, -1.0)), ("F", (-1.0, -1.0) ./ √2), ("G", (-1.0, 0.0)), ("H", (-1.0, 1.0) ./ √2))]

thule_profiles() = [("CapA", (-390e3, 0.0), (-590e3, 450e3), 10e3), ("CapB", (390e3, 0.0), (590e3, 450e3), 10e3),
                    ("CapC", (-390e3, 0.0), (-590e3, -450e3), 10e3), ("CapD", (390e3, 0.0), (590e3, -450e3), 10e3),
                    ("HalA", (-150e3, 0.0), (-150e3, 740e3), 10e3), ("HalB", (150e3, 0.0), (150e3, 740e3), 10e3),
                    ("HalC", (-150e3, 0.0), (-150e3, -740e3), 10e3), ("HalD", (150e3, 0.0), (150e3, -740e3), 10e3)]

calvingmip_profiles(experiment) = experiment in (1, 2) ? circular_profiles() : thule_profiles()

function read_grid(file)
    return jldopen(file) do f
        Hx, Hy, Nx, Ny = f["grid/Hx"], f["grid/Hy"], f["grid/Nx"], f["grid/Ny"]
        x, y = collect(f["grid/xᶜᵃᵃ"][Hx+1:Hx+Nx]), collect(f["grid/yᵃᶜᵃ"][Hy+1:Hy+Ny])
        return (; x, y, Nx, Ny, Hx, Hy, Δx = x[2] - x[1], Δy = y[2] - y[1])
    end
end

# A partial cell covers H / Hʳ of its area, and holds Hʳ-thick ice where it is covered.
cell_coverage(μ, H, Hʳ) = ifelse(μ > 0, 1.0, ifelse(Hʳ > 0, min(H / max(Hʳ, eps()), 1.0), 0.0))

function read_snapshot(f, n, g)
    i, j = g.Hx+1:g.Hx+g.Nx, g.Hy+1:g.Hy+g.Ny
    field(name) = Float64.(f["timeseries/$name/$n"])
    H, μ, φ, Hʳ = field("H")[i, j, 1], field("μ")[i, j, 1], field("φ")[i, j, 1], field("Hʳ")[i, j, 1]
    u, v = field("u"), field("v")
    calved = sum(field("C")[i, j, 1]) * g.Δx * g.Δy
    c = cell_coverage.(μ, H, Hʳ)
    h = ifelse.(μ .> 0, H, Hʳ)
    m = ifelse.(c .< 1 / 2, 3, ifelse.(φ .≥ 1 / 2, 1, 2))
    uᶠ, vᶠ = u[g.Hx+1:g.Hx+g.Nx+1, j, 1] .* year, v[i, g.Hy+1:g.Hy+g.Ny+1, 1] .* year
    uᶜ = (uᶠ[1:end-1, :] .+ uᶠ[2:end, :]) ./ 2
    vᶜ = (vᶠ[:, 1:end-1] .+ vᶠ[:, 2:end]) ./ 2
    return (; t = Float64(f["timeseries/t/$n"]), H, φ, c, h, m, u = uᶜ, v = vᶜ, uᶠ, vᶠ, calved)
end

# The calving flux (kg/a) at each output is the calved volume since the previous output of the same file, so restarts
# that reset the calved thickness do not count; `last_only` reads the previous output too for that difference.
function read_snapshots(file, g; last_only = false)
    return jldopen(file) do f
        iterations = sort(parse.(Int, keys(f["timeseries/t"])))
        snapshots = [read_snapshot(f, n, g) for n in (last_only ? iterations[max(end - 1, 1):end] : iterations)]
        flux = [n == 1 ? NaN : ρᵢ * (snapshots[n].calved - snapshots[n - 1].calved) / ((snapshots[n].t - snapshots[n - 1].t) / year)
                for n in eachindex(snapshots)]
        joined = [merge(s, (; calving = F)) for (s, F) in zip(snapshots, flux)]
        return last_only ? joined[end:end] : joined
    end
end

# Flux (kg/a) through the faces between grounded and floating ice.
function grounding_line_flux(s, g)
    flux = 0.0
    for j in 1:g.Ny, i in 1:g.Nx
        s.c[i, j] ≥ 1 / 2 && s.φ[i, j] ≥ 1 / 2 || continue
        faces = ((i - 1, j, -s.uᶠ[i, j], g.Δy), (i + 1, j, s.uᶠ[i + 1, j], g.Δy),
                 (i, j - 1, -s.vᶠ[i, j], g.Δx), (i, j + 1, s.vᶠ[i, j + 1], g.Δx))
        for (i′, j′, uₙ, Δ) in faces
            1 ≤ i′ ≤ g.Nx && 1 ≤ j′ ≤ g.Ny && s.c[i′, j′] ≥ 1 / 2 && s.φ[i′, j′] < 1 / 2 || continue
            flux += ρᵢ * (s.h[i, j] * max(uₙ, 0.0) + s.h[i′, j′] * min(uₙ, 0.0)) * Δ
        end
    end
    return flux
end

function scalars(s, g, b)
    A = g.Δx * g.Δy
    grounded = s.φ .≥ 1 / 2
    flotation = max.(.-b, 0) .* (ρₒ / ρᵢ)
    quadrant(sx, sy) = sum(s.c[i, j] for i in 1:g.Nx, j in 1:g.Ny if sign(g.x[i]) == sx && sign(g.y[j]) == sy) * A
    return (; iareagr = sum(s.c .* grounded) * A,
              iareafl = sum(s.c .* .!grounded) * A,
              lim = ρᵢ * sum(s.H) * A,
              limnsw = ρᵢ * sum(grounded .* s.c .* max.(s.h .- flotation, 0)) * A,
              tendlicalvf = s.calving, tendligroundf = grounding_line_flux(s, g),
              iareatotalNW = quadrant(-1, 1), iareatotalNE = quadrant(1, 1),
              iareatotalSW = quadrant(-1, -1), iareatotalSE = quadrant(1, -1))
end

function bilinear(F, xₚ, yₚ, g)
    i = clamp(floor(Int, (xₚ - g.x[1]) / g.Δx) + 1, 1, g.Nx - 1)
    j = clamp(floor(Int, (yₚ - g.y[1]) / g.Δy) + 1, 1, g.Ny - 1)
    α, β = (xₚ - g.x[i]) / g.Δx, (yₚ - g.y[j]) / g.Δy
    return (1 - α) * (1 - β) * F[i, j] + α * (1 - β) * F[i + 1, j] + (1 - α) * β * F[i, j + 1] + α * β * F[i + 1, j + 1]
end

nearest(F, xₚ, yₚ, g) = F[clamp(round(Int, (xₚ - g.x[1]) / g.Δx) + 1, 1, g.Nx), clamp(round(Int, (yₚ - g.y[1]) / g.Δy) + 1, 1, g.Ny)]

function profile(s, start, stop, spacing, g)
    L = hypot((stop .- start)...)
    distance = collect(0:spacing:L)
    points = [start .+ (d / L) .* (stop .- start) for d in distance]
    pm = [nearest(s.m, p..., g) for p in points]
    nanocean(F) = [ifelse(m == 3, NaN, bilinear(F, p..., g)) for (m, p) in zip(pm, points)]
    pc = [bilinear(s.c, p..., g) for p in points]
    # The front is where the covered fraction falls through one half, past the last ice point.
    k = findfirst(n -> pc[n] ≥ 1 / 2 && pc[n + 1] < 1 / 2, 1:length(points) - 1)
    front = if isnothing(k)
        (NaN, NaN, NaN, NaN, NaN)
    else
        d = distance[k] + (pc[k] - 1 / 2) / (pc[k] - pc[k + 1]) * spacing
        xf, yf = start .+ (d / L) .* (stop .- start)
        (xf, yf, nearest(s.u, points[k]..., g), nearest(s.v, points[k]..., g), nearest(s.h, points[k]..., g))
    end
    return (; s = distance, lithk = nanocean(s.h), xvelmean = nanocean(s.u), yvelmean = nanocean(s.v), mask = pm, front)
end

# Exp2 and Exp4 start from the last output of their spin-up, and Exp4 joins its two runs.
function calvingmip_field_files(experiment, Δx)
    experiment in (1, 3) && return [calvingmip_name(experiment, Δx) * "_fields.jld2"]
    spinup = calvingmip_name(experiment - 1, Δx) * "_fields.jld2"
    runs = experiment == 2 ? [calvingmip_name(2, Δx)] : [calvingmip_name(4, Δx), calvingmip_name(4, Δx) * "_free"]
    return [spinup; runs .* "_fields.jld2"]
end

function calvingmip_netcdf(experiment; Δx = 5kilometers, institution = "TEST")
    field_files = calvingmip_field_files(experiment, Δx)
    bed = calvingmip_bed(experiment)
    profiles = calvingmip_profiles(experiment)

    g = read_grid(first(field_files))
    snapshots = reduce(vcat, [read_snapshots(file, g; last_only = n == 1 && length(field_files) > 1)
                              for (n, file) in enumerate(field_files)])
    unique!(s -> s.t, snapshots)

    b = [bed(xᵢ, yⱼ) for xᵢ in g.x, yⱼ in g.y]
    t₀ = first(snapshots).t
    times = [(s.t - t₀) / year for s in snapshots]
    time100 = findall(t -> isapprox(t, round(t / 100) * 100; atol = 1e-6), times)
    # An output without a calved difference (the first of each file) takes the next one's.
    for n in reverse(1:length(snapshots) - 1)
        isnan(snapshots[n].calving) && (snapshots[n] = merge(snapshots[n], (; calving = snapshots[n + 1].calving)))
    end
    series = [scalars(s, g, b) for s in snapshots]
    lines = Dict(name => [profile(s, start, stop, spacing, g) for s in snapshots] for (name, start, stop, spacing) in profiles)

    filename = "CalvingMIP_EXP$(experiment)_Nunatak_$(institution).nc"
    isfile(filename) && rm(filename)
    NCDataset(filename, "c") do ds
        defDim(ds, "X", g.Nx)
        defDim(ds, "Y", g.Ny)
        defDim(ds, "Time1", length(times))
        defDim(ds, "Time100", length(time100))
        defVar(ds, "X", g.x, ("X",); attrib = ["units" => "m"])
        defVar(ds, "Y", g.y, ("Y",); attrib = ["units" => "m"])
        defVar(ds, "Time1", times, ("Time1",); attrib = ["units" => "a"])
        defVar(ds, "Time100", times[time100], ("Time100",); attrib = ["units" => "a"])

        snapshot(f) = cat((f(snapshots[n]) for n in time100)...; dims = 3)
        oceanNaN(F, s) = ifelse.(s.m .== 3, NaN, F)
        defVar(ds, "lithk", snapshot(s -> oceanNaN(s.h, s)), ("X", "Y", "Time100");
               attrib = ["units" => "m", "standard_name" => "land_ice_thickness"])
        defVar(ds, "xvelmean", snapshot(s -> oceanNaN(s.u, s)), ("X", "Y", "Time100");
               attrib = ["units" => "m a-1", "standard_name" => "land_ice_vertical_mean_x_velocity"])
        defVar(ds, "yvelmean", snapshot(s -> oceanNaN(s.v, s)), ("X", "Y", "Time100");
               attrib = ["units" => "m a-1", "standard_name" => "land_ice_vertical_mean_y_velocity"])
        defVar(ds, "mask", snapshot(s -> Float64.(s.m)), ("X", "Y", "Time100");
               attrib = ["flag_values" => "1 2 3", "flag_meanings" => "grounded floating open_ocean"])
        defVar(ds, "topg", b, ("X", "Y"); attrib = ["units" => "m", "standard_name" => "bedrock_altimetry"])

        for (name, units) in (("iareagr", "m2"), ("iareafl", "m2"), ("lim", "kg"), ("limnsw", "kg"), ("tendlicalvf", "kg a-1"),
                              ("tendligroundf", "kg a-1"),
                              ("iareatotalNW", "m2"), ("iareatotalNE", "m2"), ("iareatotalSW", "m2"), ("iareatotalSE", "m2"))
            defVar(ds, name, [getproperty(q, Symbol(name)) for q in series], ("Time1",); attrib = ["units" => units])
        end

        for (name, _, _, _) in profiles
            ls = lines[name]
            dim = "s" * name
            defDim(ds, dim, length(first(ls).s))
            defVar(ds, "s" * name, first(ls).s, (dim,); attrib = ["units" => "m"])
            along(f) = reduce(hcat, (f(l) for l in ls))
            for (field, units) in (("lithk", "m"), ("xvelmean", "m a-1"), ("yvelmean", "m a-1"))
                defVar(ds, field * name, along(l -> getproperty(l, Symbol(field))), (dim, "Time1"); attrib = ["units" => units])
            end
            defVar(ds, "mask" * name, along(l -> Float64.(l.mask)), (dim, "Time1"))
            for (n, (field, units)) in enumerate((("xcf", "m"), ("ycf", "m"), ("xvelmeancf", "m a-1"), ("yvelmeancf", "m a-1"), ("lithkcf", "m")))
                defVar(ds, field * name, [l.front[n] for l in ls], ("Time1",); attrib = ["units" => units])
            end
        end
    end

    @info @sprintf("Wrote %s: %d yearly and %d 100-year outputs from %.0f to %.0f a", filename, length(times),
                   length(time100), first(times), last(times))
    return filename
end

experiment_number(filename) = parse(Int, match(r"EXP(\d)", filename)[1])

# Radial front displacement (km), the direction the prescribed front velocity acts in, from the initial front or, on
# profiles that end inside it, from the 750 km maximum extent.
function front_displacement(ds, name)
    r = hypot.(Array(ds["xcf" * name]), Array(ds["ycf" * name])) ./ 1e3
    r₀ = isnan(r[1]) ? front_radius / kilometers : r[1]
    return r .- r₀
end

# Front displacement (km) from Wv = -Wᵐᵃˣ sin(2π t / 1000 a), integrated; Exp4 prescribes it for the first 500 a only.
prescribed_displacement(experiment, t) = -(experiment == 2 ? 300 : 750) * 1000 / 2π * (1 - cos(2π * min(t, experiment == 4 ? 500 : t) / 1000)) / 1e3

function plot_calvingmip(filename)
    experiment = experiment_number(filename)
    names = first.(calvingmip_profiles(experiment))
    ds = NCDataset(filename)
    t = Array(ds["Time1"])
    t100 = Array(ds["Time100"])
    x, y = Array(ds["X"]) ./ 1e3, Array(ds["Y"]) ./ 1e3

    fig = Figure(size = (1400, 1000))

    ax = Axis(fig[1, 1]; xlabel = "time (a)", ylabel = "front displacement (km)", title = "Front position along the profiles")
    for name in names
        lines!(ax, t, front_displacement(ds, name); label = name)
    end
    if experiment in (2, 4)
        tᵖ = experiment == 4 ? filter(≤(500), t) : t
        lines!(ax, tᵖ, prescribed_displacement.(experiment, tᵖ); color = :black, linestyle = :dash, linewidth = 3, label = "prescribed")
    end
    axislegend(ax; position = :lb, nbanks = 2)

    ax = Axis(fig[1, 2]; xlabel = "time (a)", ylabel = "area (10³ km²)", title = "Ice area")
    lines!(ax, t, Array(ds["iareagr"]) ./ 1e9; label = "grounded")
    lines!(ax, t, Array(ds["iareafl"]) ./ 1e9; label = "floating")
    axislegend(ax; position = :rc)

    ax = Axis(fig[2, 1]; xlabel = "time (a)", ylabel = "mass (10¹⁵ kg)", title = "Ice mass and calving flux")
    lines!(ax, t, Array(ds["lim"]) ./ 1e15; label = "total")
    lines!(ax, t, Array(ds["limnsw"]) ./ 1e15; label = "above flotation")
    ax′ = Axis(fig[2, 1]; ylabel = "calving flux (10¹² kg/a)", yaxisposition = :right)
    hidespines!(ax′)
    hidexdecorations!(ax′)
    lines!(ax′, t, Array(ds["tendlicalvf"]) ./ 1e12; color = :gray, linestyle = :dot)
    axislegend(ax; position = :lt)

    ax = Axis(fig[2, 2]; aspect = DataAspect(), xlabel = "x (km)", ylabel = "y (km)",
              title = @sprintf("Thickness at %.0f a, grounding line in white", t100[end]))
    hm = heatmap!(ax, x, y, Array(ds["lithk"][:, :, end]); colormap = :ice, nan_color = :lightgray)
    contour!(ax, x, y, Float64.(Array(ds["mask"][:, :, end]) .== 1); levels = [0.5], color = :white, linewidth = 1.5)
    Colorbar(fig[2, 3], hm; label = "thickness (m)")

    Label(fig[0, :], basename(filename); fontsize = 20)
    close(ds)

    output = replace(filename, ".nc" => ".png")
    save(output, fig)
    return output
end

# Mean, minimum and maximum front displacement over the profiles against the prescribed one, at `times` (a).
function calvingmip_summary(filename; times = 0:250:1000)
    experiment = experiment_number(filename)
    ds = NCDataset(filename)
    t = Array(ds["Time1"])
    displacements = [front_displacement(ds, name) for name in first.(calvingmip_profiles(experiment))]
    lim = Array(ds["lim"])
    close(ds)
    for tₙ in times
        n = argmin(abs.(t .- tₙ))
        d = filter(!isnan, [dᵖ[n] for dᵖ in displacements])
        isempty(d) && continue
        @info @sprintf("Exp%d t = %4.0f a: front displacement mean %7.1f, min %7.1f, max %7.1f km, prescribed %7.1f km, mass %.4e kg",
                       experiment, t[n], sum(d) / length(d), minimum(d), maximum(d), prescribed_displacement(experiment, t[n]), lim[n])
    end
    return nothing
end
