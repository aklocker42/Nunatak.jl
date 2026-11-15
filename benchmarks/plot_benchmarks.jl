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

function plot_benchmarks(dir = pwd())
    mkpath(figures)
    for (file, plot) in (("halfar_timeseries.csv", plot_halfar), ("halfpipe_streamice_timeseries.csv", plot_halfpipe),
                         ("eismint2_a_timeseries.csv", plot_eismint2_a), ("mismip_plus_timeseries.csv", plot_mismip_plus))
        isfile(joinpath(dir, file)) ? plot(dir) : @info "No $(file) in $(dir); skipping."
    end
    return nothing
end
