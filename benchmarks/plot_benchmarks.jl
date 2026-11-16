# Figures for the documentation from the CSV files the benchmarks write, saved to docs/src/assets/figures.
# Needs an environment with CairoMakie and DelimitedFiles. Run from the directory holding the benchmark output:
#
#     include("plot_benchmarks.jl"); plot_benchmarks()
using CairoMakie, DelimitedFiles

figures = joinpath(@__DIR__, "..", "docs", "src", "assets", "figures")

function read_table(file)
    data, header = readdlm(file, ','; header = true)
    return Dict(String(h) => data[:, k] for (k, h) in enumerate(vec(header)))
end

function plot_halfar(dir; Δx = 25e3)
    H = readdlm(joinpath(dir, "halfar_thickness.csv"), ',')
    Hₑ = readdlm(joinpath(dir, "halfar_exact_thickness.csv"), ',')
    series = read_table(joinpath(dir, "halfar_timeseries.csv"))
    Nx = size(H, 1)
    x = ((1:Nx) .- (Nx + 1) / 2) .* Δx ./ 1e3
    j = (size(H, 2) + 1) ÷ 2

    fig = Figure(size = (900, 360))
    ax = Axis(fig[1, 1]; xlabel = "x (km)", ylabel = "thickness (m)", title = "Centre section at 25 ka")
    lines!(ax, x, Hₑ[:, j]; label = "Halfar (1983)", color = :black)
    scatter!(ax, x, H[:, j]; label = "Nunatak", markersize = 6)
    axislegend(ax; position = :rb)
    ax = Axis(fig[1, 2]; xlabel = "t − t₀ (a)", ylabel = "error (m)", title = "Thickness error")
    lines!(ax, series["t_minus_t0_years"], series["max_abs_error_m"]; label = "max |ΔH|")
    lines!(ax, series["t_minus_t0_years"], series["mean_abs_error_m"]; label = "mean |ΔH|")
    lines!(ax, series["t_minus_t0_years"], abs.(series["divide_error_m"]); label = "|ΔH| at the divide")
    axislegend(ax; position = :lt)
    save(joinpath(figures, "halfar_section_and_error.png"), fig)
    return fig
end

function plot_halfpipe(dir)
    nunatak = read_table(joinpath(dir, "halfpipe_streamice_timeseries.csv"))
    streamice = read_table(joinpath(@__DIR__, "halfpipe_streamice_reference.csv"))

    fig = Figure(size = (900, 360))
    for (k, (name, label)) in enumerate(("volume_km3" => "volume (km³)", "centerline_front_km" => "centerline front (km)"))
        ax = Axis(fig[1, k]; xlabel = "t (a)", ylabel = label)
        lines!(ax, streamice["t_years"], streamice[name]; label = "STREAMICE", color = :black)
        scatter!(ax, nunatak["t_years"], nunatak[name]; label = "Nunatak", markersize = 8)
        k == 1 && axislegend(ax; position = :lt)
    end
    save(joinpath(figures, "halfpipe_streamice_volume_and_front.png"), fig)
    return fig
end

function plot_eismint2_a(dir; output = "eismint2_a", Δx = 25e3)
    series = read_table(joinpath(dir, output * "_timeseries.csv"))
    H = readdlm(joinpath(dir, output * "_thickness.csv"), ',')
    x = ((1:size(H, 1)) .- 1) .* Δx ./ 1e3

    fig = Figure(size = (1200, 360))
    ax = Axis(fig[1, 1]; xlabel = "t (ka)", ylabel = "volume (10⁶ km³)")
    lines!(ax, series["t_years"] ./ 1e3, series["volume_1e6km3"])
    ax = Axis(fig[1, 2]; xlabel = "t (ka)", ylabel = "divide basal temperature (K)")
    lines!(ax, series["t_years"] ./ 1e3, series["divide_basal_T_K"])
    ax = Axis(fig[1, 3]; xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect(), title = "Thickness at 200 ka")
    hm = heatmap!(ax, x, x, replace(H, 0.0 => NaN); colormap = :ice)
    Colorbar(fig[1, 4], hm; label = "m")
    save(joinpath(figures, "eismint2_a_evolution.png"), fig)
    return fig
end

function plot_mismip_plus(dir; output = "mismip_plus")
    data, header = readdlm(joinpath(dir, output * "_timeseries.csv"), ','; header = true)
    reference, _ = readdlm(joinpath(@__DIR__, "mismip_plus_streamice_reference.csv"), ','; header = true)
    experiments = ("Ice1r", "Ice1ra", "Ice2r", "Ice2ra")

    fig = Figure(size = (900, 360))
    for (k, (name, label, scale)) in enumerate((("centerline_grounding_line_m", "centerline grounding line (km)", 1e-3),
                                                ("volume_above_flotation_m3", "volume above flotation (10³ km³)", 1e-12)))
        c = findfirst(==(name), vec(header))
        ax = Axis(fig[1, k]; xlabel = "t (a)", ylabel = label)
        for (e, experiment) in enumerate(experiments)
            color = Makie.wong_colors()[e]
            rows = reference[:, 1] .== experiment
            lines!(ax, Float64.(reference[rows, 2]), Float64.(reference[rows, c]) .* scale; color, linestyle = :dash)
            rows = data[:, 1] .== experiment
            lines!(ax, Float64.(data[rows, 2]), Float64.(data[rows, c]) .* scale; color, label = experiment)
        end
        k == 1 && axislegend(ax; position = :lb)
    end
    Label(fig[2, 1:2], "solid: Nunatak, dashed: STREAMICE (Cornford et al. 2020)"; fontsize = 12)
    save(joinpath(figures, "mismip_plus_grounding_line_and_vaf.png"), fig)
    return fig
end

function plot_halfpipe_adjoint(dir; output = "halfpipe_streamice_adjoint")
    g = read_table(joinpath(dir, output * "_gradient.csv"))
    i = g["i"]

    fig = Figure(size = (900, 360))
    ax = Axis(fig[1, 1]; xlabel = "cell i along the centerline", ylabel = "∂J/∂(basal melt) (a/m)",
              title = "Gradient at the check cells")
    scatter!(ax, i, g["streamice_adjoint"]; label = "STREAMICE adjoint", color = :black, marker = :rect, markersize = 14)
    scatter!(ax, i, g["streamice_finite_difference"]; label = "STREAMICE finite difference", color = :gray, marker = :cross, markersize = 14)
    scatter!(ax, i, g["nunatak_adjoint"]; label = "Nunatak adjoint", markersize = 10)
    scatter!(ax, i, g["nunatak_finite_difference"]; label = "Nunatak finite difference", marker = :xcross, markersize = 12)
    axislegend(ax; position = :lb)
    ax = Axis(fig[1, 2]; xlabel = "cell i along the centerline", ylabel = "relative difference", title = "Consistency")
    scatter!(ax, i, g["nunatak_adjoint"] ./ g["nunatak_finite_difference"] .- 1; label = "Nunatak adjoint / finite difference − 1")
    scatter!(ax, i, g["nunatak_adjoint"] ./ g["streamice_adjoint"] .- 1; label = "Nunatak / STREAMICE adjoint − 1", marker = :rect)
    axislegend(ax; position = :rt)
    save(joinpath(figures, output * "_gradient.png"), fig)
    return fig
end

function plot_kleiner2015_a(dir; output = "kleiner2015_a")
    s = read_table(joinpath(dir, output * "_timeseries.csv"))
    t = s["t_years"] ./ 1e3
    cooling = 150 .<= t .<= 160

    fig = Figure(size = (900, 360))
    ax = Axis(fig[1, 1]; xlabel = "t (ka)", ylabel = "basal melt (m/a w.e.)", title = "Melt rate after the surface cooling at 150 ka")
    lines!(ax, t[cooling], s["melt_exact_we_m_per_a"][cooling]; label = "analytic", color = :black)
    lines!(ax, t[cooling], s["melt_we_m_per_a"][cooling]; label = "Nunatak", linestyle = :dash)
    axislegend(ax; position = :rt)
    ax = Axis(fig[1, 2]; xlabel = "t (ka)", ylabel = "basal water (m w.e.)", title = "Basal water layer over the cycle")
    lines!(ax, t, s["water_we_m"])
    save(joinpath(figures, output * "_melt_and_water.png"), fig)
    return fig
end

function plot_kleiner2015_b(dir; output = "kleiner2015_b")
    p = read_table(joinpath(dir, output * "_profile.csv"))
    table = read_table(joinpath(dir, output * "_transition.csv"))
    analytic_height = table["transition_height_m"][1] - table["height_error_m"][1]

    fig = Figure(size = (900, 360))
    ax = Axis(fig[1, 1]; xlabel = "enthalpy (kJ/kg)", ylabel = "ζ", title = "Steady profile, Nσ = 100, ratio 10⁻³")
    lines!(ax, p["enthalpy_exact_J_per_kg"] ./ 1e3, p["zeta"]; label = "analytic", color = :black)
    scatter!(ax, p["enthalpy_J_per_kg"] ./ 1e3, p["zeta"]; label = "Nunatak", markersize = 5)
    axislegend(ax; position = :rt)
    ax = Axis(fig[1, 2]; xlabel = "Nσ", ylabel = "cold-temperate transition height (m)", xscale = log10, title = "Transition height")
    hlines!(ax, [analytic_height]; label = "analytic", color = :black)
    for ratio in unique(table["diffusivity_ratio"])
        rows = table["diffusivity_ratio"] .== ratio
        scatterlines!(ax, table["Nsigma"][rows], table["transition_height_m"][rows]; label = "ratio $(ratio)")
    end
    axislegend(ax; position = :rt)
    save(joinpath(figures, output * "_profile_and_transition.png"), fig)
    return fig
end

function plot_benchmarks(dir = pwd())
    mkpath(figures)
    for (file, plot) in (("halfar_timeseries.csv", plot_halfar), ("halfpipe_streamice_timeseries.csv", plot_halfpipe),
                         ("eismint2_a_timeseries.csv", plot_eismint2_a), ("mismip_plus_timeseries.csv", plot_mismip_plus),
                         ("halfpipe_streamice_adjoint_gradient.csv", plot_halfpipe_adjoint),
                         ("kleiner2015_a_timeseries.csv", plot_kleiner2015_a), ("kleiner2015_b_profile.csv", plot_kleiner2015_b))
        isfile(joinpath(dir, file)) ? plot(dir) : @info "No $(file) in $(dir); skipping."
    end
    return nothing
end
