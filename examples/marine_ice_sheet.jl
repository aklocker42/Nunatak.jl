# # A marine ice sheet advancing into the ocean
#
# This example grows a marine ice sheet in a channel whose bed lies below sea level, the geometry of the MISMIP+
# intercomparison. Ice starts as a thin slab, thickens under steady accumulation, and its grounding line, where
# the ice begins to float, advances down the channel. The example shows how to
#
#   * set a bed elevation below sea level and let the model diagnose where the ice floats,
#   * solve the depth-integrated stress balance with a regularized Coulomb sliding law,
#   * remove ice at a calving front with a basal melt rate,
#   * record a diagnostic, the grounding-line position, with a callback.
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
using Nunatak
using CairoMakie
using Printf

year = years = 31556926seconds

# ## The grid
#
# The channel is 640 km long and 80 km wide, with one extra column of cells beyond its end where the ice is
# removed. We use 8 km cells to keep the example quick; MISMIP+ asks for 1 km or finer.

Δx = 8kilometers
Lx, Ly = 640kilometers, 80kilometers

grid = RectilinearGrid(CPU();
                       size = (Int(Lx / Δx) + 1, Int(Ly / Δx)),
                       halo = (4, 4),
                       x = (0, Lx + Δx),
                       y = (0, Ly),
                       topology = (Bounded, Bounded, Flat))

# ## The bed
#
# The MISMIP+ bed deepens downstream with a sill near 300 km, and steep side walls confine the ice to a trough
# about 50 km wide (Asay-Davis et al., 2016, Eqs. 1-4).

B₀, B₂, B₄, B₆, x̄ = -150, -728.8, 343.91, -50.57, 300kilometers
fᶜ, dᶜ, wᶜ = 4kilometers, 500, 24kilometers

along_flow_bed(x) = B₀ + B₂ * (x / x̄)^2 + B₄ * (x / x̄)^4 + B₆ * (x / x̄)^6
across_flow_bed(y) = dᶜ / (1 + exp(-2 * (y - Ly / 2 - wᶜ) / fᶜ)) + dᶜ / (1 + exp(2 * (y - Ly / 2 + wᶜ) / fᶜ))
bed(x, y) = max(along_flow_bed(x) + across_flow_bed(y), -720)

# ## The model
#
# The ice flows by the depth-integrated viscosity approximation (DIVA). The sliding law of Schoof (2005) reduces to
# a Weertman law where the ice is well grounded and to Coulomb friction near the grounding line, where the effective
# pressure falls to zero. `SubgridGroundedFraction` finds the grounding line within cells. Ice past the end of the
# channel melts at 100 km/a, which acts as a calving front at x = 640 km.

rheology = GlenFlowLaw(grid; rate_factor = 2.0392e-17 / year, strain_rate_regularization = 1e-5 / year)

friction = RegularizedCoulombFriction(grid; coefficient = 1e4 * year^(1 / 3), coulomb_coefficient = 0.5,
                                      exponent = 1 / 3, velocity_regularization = 1e-5 / year)

# The velocity changes little within a time step, so two Picard iterations per step, starting from the previous
# velocity, are enough.

velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-8, maxiter = 20_000),
                                      picard_iteration_control = FixedIterations(2))

model = IceSheetModel(grid; rheology, friction, velocity_solver, bed,
                      stress_balance = DepthIntegrated(shear = DIVAShear(), side_walls = FreeSlipWalls()),
                      grounding = SubgridGroundedFraction(),
                      constants = IceSheetConstants(grid; ice_density = 918, ocean_density = 1028),
                      surface_mass_balance = 0.3 / year,
                      basal_melt_rate = (x, y) -> ifelse(x > Lx, 1e5 / year, 0.0))

set!(model, H = (x, y) -> ifelse(x < Lx, 100.0, 0.0))

# ## Tracking the grounding line
#
# The grounded fraction `φ` is one where the ice rests on the bed and zero where it floats. Summed along the
# centre of the channel it gives the grounded length, which is the grounding-line position.

times = Float64[]
grounding_line = Float64[]

function record_grounding_line(sim)
    φ = Array(interior(sim.model.geometry.φ, :, grid.Ny ÷ 2, 1))
    push!(times, time(sim))
    push!(grounding_line, sum(φ) * Δx)
    @info @sprintf("t = %5.0f a, grounding line at %5.1f km, max speed = %6.1f m/a, wall time: %s",
                   time(sim) / year, grounding_line[end] / kilometer, maximum(abs, sim.model.velocities.u) * year,
                   prettytime(sim.run_wall_time))
    return nothing
end

simulation = Simulation(model; Δt = 0.5years, stop_time = 5000years)
add_callback!(simulation, record_grounding_line, TimeInterval(100years))

run!(simulation)

# ## Plotting
#
# The grounding line advances quickly at first and slows as the ice sheet approaches a steady state. MISMIP+
# models settle with the grounding line at about 450 km after 20 000 years or more.

x, y = xnodes(grid, Center()), ynodes(grid, Center())
H = Array(interior(model.geometry.H, :, :, 1))
b = Array(interior(model.bed, :, :, 1))
s = Array(interior(model.geometry.s, :, :, 1))
j = grid.Ny ÷ 2

fig = Figure(size = (1000, 700))

ax = Axis(fig[1, 1]; xlabel = "t (a)", ylabel = "grounding line (km)", title = "Grounding-line advance")
lines!(ax, times ./ year, grounding_line ./ kilometer)

ax = Axis(fig[1, 2]; xlabel = "x (km)", ylabel = "z (m)", title = @sprintf("Centreline at %.0f a", time(simulation) / year))
band!(ax, x ./ kilometer, s[:, j] .- H[:, j], s[:, j]; color = :lightblue, label = "ice")
lines!(ax, x ./ kilometer, b[:, j]; color = :brown, label = "bed")
hlines!(ax, 0; color = :blue, linestyle = :dash, label = "sea level")
axislegend(ax; position = :lb)

xᶠ = xnodes(grid, Face())
u = Array(interior(model.velocities.u, :, :, 1))
ax = Axis(fig[2, 1:2]; xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect(), title = "Along-flow speed (m/a)")
hm = heatmap!(ax, xᶠ ./ kilometer, y ./ kilometer, u .* year; colormap = :viridis)
contour!(ax, x ./ kilometer, y ./ kilometer, Array(interior(model.geometry.φ, :, :, 1)); levels = [0.5], color = :white)
Colorbar(fig[3, 1:2], hm; vertical = false)

save("marine_ice_sheet.png", fig)
