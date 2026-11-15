# # Growing an ice cap
#
# This example grows an ice cap from bare ground on a flat bed. Snow accumulates near the centre of the domain
# and melts away from it, and the ice spreads under its own weight following the shallow-ice approximation.
# The example shows how to
#
#   * build an `IceSheetModel` with the `ShallowIce` stress balance,
#   * prescribe a surface mass balance as a function of position,
#   * adapt the time step to the diffusive stability limit of shallow ice,
#   * save and plot the evolving thickness.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# pkg"add https://github.com/aklocker42/Nunatak.jl"
# ```

using Oceananigans
using Oceananigans.Units
using Oceananigans.Diagnostics: cell_diffusion_timescale
using Nunatak
using CairoMakie
using Printf

# `Oceananigans.Units` has no year, so we define one, with the length used by the EISMINT and MISMIP+ intercomparisons.

year = years = 31556926seconds

# ## The grid
#
# Ice-sheet grids are two-dimensional: the vertical direction is `Flat`, and Nunatak builds a terrain-following
# grid through the ice itself. We use a 1500 km square with 25 km cells.

grid = RectilinearGrid(CPU();
                       size = (60, 60),
                       halo = (4, 4),
                       x = (-750kilometers, 750kilometers),
                       y = (-750kilometers, 750kilometers),
                       topology = (Bounded, Bounded, Flat))

# ## Surface mass balance
#
# As in EISMINT II, the mass balance is 0.5 m/a of ice near the centre and decreases by 1 cm/a per kilometre beyond
# 400 km from it, so the ice cap cannot grow past about 450 km.

distance(x, y) = sqrt(x^2 + y^2) / kilometer
surface_mass_balance(x, y) = min(0.5, 1e-2 * (450 - distance(x, y))) / year

# ## The model
#
# Glen's flow law with a constant rate factor makes the ice isothermal. Without a friction law the ice is frozen
# to its bed.

rheology = GlenFlowLaw(grid; rate_factor = 1e-16 / year, exponent = 3)

model = IceSheetModel(grid; rheology, stress_balance = ShallowIce(), surface_mass_balance)

# ## The simulation
#
# Shallow-ice flow is a nonlinear diffusion of the thickness, so explicit time steps must stay below the
# diffusive limit. `cell_diffusion_timescale` computes that limit from the current geometry, and a
# `TimeStepWizard` keeps the step at a tenth of it; a quarter goes unstable once the cap is about 1800 m thick.

simulation = Simulation(model; Δt = 1year, stop_time = 20_000years)

conjure_time_step_wizard!(simulation, IterationInterval(10); diffusive_cfl = 0.1, cell_diffusion_timescale,
                          max_Δt = 50years)

function progress(sim)
    H = sim.model.geometry.H
    volume = sum(H) * 25kilometers^2
    @info @sprintf("t = %6.0f a, Δt = %5.2f a, max thickness = %6.1f m, volume = %.3e km³, wall time: %s",
                   time(sim) / year, sim.Δt / year, maximum(H), volume / kilometer^3, prettytime(sim.run_wall_time))
    return nothing
end

add_callback!(simulation, progress, TimeInterval(1000years))

simulation.output_writers[:thickness] = JLD2Writer(model, (; H = model.geometry.H);
                                                   filename = "ice_cap.jld2",
                                                   schedule = TimeInterval(1000years),
                                                   overwrite_files = true)

run!(simulation)

# ## Plotting the ice cap
#
# We load the saved thickness as a `FieldTimeSeries` and plot sections through the centre at several times, next
# to the final thickness.

H = FieldTimeSeries("ice_cap.jld2", "H")
times = H.times
x, y = xnodes(grid, Center()), ynodes(grid, Center())
j = grid.Ny ÷ 2

fig = Figure(size = (1000, 400))

ax = Axis(fig[1, 1]; xlabel = "x (km)", ylabel = "thickness (m)", title = "Section through the centre")
for n in (2, 3, 5, 9, length(times))
    lines!(ax, x ./ kilometer, interior(H[n], :, j, 1); label = @sprintf("%.0f ka", times[n] / 1000years))
end
axislegend(ax; position = :rt)

ax = Axis(fig[1, 2]; xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect(),
          title = @sprintf("Thickness at %.0f ka", times[end] / 1000years))
Hₙ = interior(H[end], :, :, 1)
hm = heatmap!(ax, x ./ kilometer, y ./ kilometer, replace(Hₙ, 0.0 => NaN); colormap = :ice)
Colorbar(fig[1, 3], hm; label = "m")

save("ice_cap.png", fig)
