# Plots the output of friction_inversion.jl from the current directory: true vs inverted γ (two λ), speeds, cost history.
# Needs an environment with CairoMakie and JLD2, which examples/Project.toml leaves out to keep the inversion light.
using CairoMakie, JLD2, Printf

dir = pwd()
runs = [load(joinpath(dir, "friction_inversion_lambda$(s).jld2")) for s in ("1e-03", "1e-05")]
sec_per_year = 3.15576e7
km(r) = (r["xᶜ"] ./ 1e3, r["yᶜ"] ./ 1e3)

masked(a, g) = ifelse.(g, a, NaN)
speed(u, v) = sqrt.(((u[1:end-1, :] .+ u[2:end, :]) ./ 2) .^ 2 .+ ((v[:, 1:end-1] .+ v[:, 2:end]) ./ 2) .^ 2) .* sec_per_year

r = runs[2]
x, y = km(r)
g = r["grounded"]
γrange = (-1, 1)

fig = Figure(size = (1500, 1150))
panels = [(r["γ_true"], "true γ"),
          (runs[1]["γ_inv"], @sprintf("inverted γ, λ = %.0e", runs[1]["λ"])),
          (r["γ_inv"], @sprintf("inverted γ, λ = %.0e", r["λ"]))]
for (n, (γ, title)) in enumerate(panels)
    ax = Axis(fig[1, n]; title, xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
    hm = heatmap!(ax, x, y, masked(γ, g); colormap = :balance, colorrange = γrange)
    contour!(ax, x, y, Float64.(g); levels = [0.5], color = :black)
    n == 3 && Colorbar(fig[1, 4], hm; label = "γ = log(C / C₀)")
end

ax = Axis(fig[2, 1]; title = "γ error, λ = 1e-05", xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
hm = heatmap!(ax, x, y, masked(r["γ_inv"] .- r["γ_true"], g); colormap = :balance, colorrange = (-0.3, 0.3))
contour!(ax, x, y, Float64.(g); levels = [0.5], color = :black)
Colorbar(fig[3, 1], hm; label = "γ error", vertical = false, flipaxis = false)

sᵒ = speed(r["uᵒ"], r["vᵒ"])
ax = Axis(fig[2, 2]; title = "observed speed (m/a)", xlabel = "x (km)", aspect = DataAspect())
hm = heatmap!(ax, x, y, log10.(max.(sᵒ, 1)); colormap = :viridis, colorrange = (1, 4.2))
Colorbar(fig[3, 2], hm; label = "log₁₀ speed (m/a)", vertical = false, flipaxis = false)
contour!(ax, x, y, Float64.(g); levels = [0.5], color = :white)

ax = Axis(fig[2, 3]; title = "inverted − observed speed (m/a)", xlabel = "x (km)", aspect = DataAspect())
hm = heatmap!(ax, x, y, speed(r["u_inv"], r["v_inv"]) .- sᵒ; colormap = :balance, colorrange = (-300, 300))
Colorbar(fig[3, 3], hm; label = "speed residual (m/a)", vertical = false, flipaxis = false)
contour!(ax, x, y, Float64.(g); levels = [0.5], color = :black)

ax = Axis(fig[4, 1:2]; title = "cost J during L-BFGS", xlabel = "evaluation", ylabel = "J", yscale = log10)
for run in runs
    lines!(ax, run["history"]; label = @sprintf("λ = %.0e", run["λ"]))
end
axislegend(ax)

j = argmin(abs.(y .- 25))
ax = Axis(fig[4, 3]; title = @sprintf("γ along y = %.0f km", y[j]), xlabel = "x (km)", ylabel = "γ")
lines!(ax, x, masked(r["γ_true"], g)[:, j]; color = :black, label = "true")
for run in runs
    lines!(ax, x, masked(run["γ_inv"], g)[:, j]; label = @sprintf("λ = %.0e", run["λ"]))
end
axislegend(ax; position = :rb)

save(joinpath(dir, "friction_inversion.png"), fig)
println("saved ", joinpath(dir, "friction_inversion.png"))
