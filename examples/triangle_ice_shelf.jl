# # An ice shelf in a bay on a triangle mesh
#
# This example solves the shallow-shelf stress balance for a floating ice shelf that fills an irregular bay.
# Instead of a rectilinear grid, the domain is covered by triangles that follow the coastline, generated with
# [Triangulate.jl](https://github.com/JuliaGeometry/Triangulate.jl). The example shows how to
#
#   * mesh a polygon with Triangulate.jl and build a `TriangleMesh` from it,
#   * constrain the velocity on the coast and let the calving front float freely,
#   * solve the stress balance with `solve_stress_balance!`,
#   * plot nodal fields on the mesh.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, Triangulate, CairoMakie"
# pkg"add https://github.com/aklocker42/Nunatak.jl"
# ```

using Oceananigans
using Oceananigans.Units
using Nunatak
using Nunatak: ice_sheet_geometry, ice_sheet_auxiliary_fields
using Triangulate
using CairoMakie
using Printf

year = years = 31556926seconds

# ## The coastline
#
# The bay is 150 km long. Its half-width grows from 40 km at the grounding line, ``x = 0``, to 50 km at the calving
# front, ``x = L``, with a wavy coast in between. We walk around the outline counter-clockwise and mark every
# boundary segment: 1 for coast and grounding line, where the ice is held still, and 2 for the calving front.

L = 150kilometers
half_width(x) = 40kilometers + 10kilometers * x / L + 5kilometers * sin(π * x / 30kilometers)

Nside = 60
xs = range(0, L, length = Nside + 1)
outline = vcat([[x, -half_width(x)] for x in xs],
               [[x, half_width(x)] for x in reverse(xs)])

Npoints = length(outline)
markers = [ifelse(n == Nside + 1, 2, 1) for n in 1:Npoints]

# Segment ``n`` joins point ``n`` to the next one around the outline; segment `Nside + 1` is the front.

triin = Triangulate.TriangulateIO()
triin.pointlist = reduce(hcat, outline)
triin.segmentlist = Int32.(reduce(hcat, [[n, mod1(n + 1, Npoints)] for n in 1:Npoints]))
triin.segmentmarkerlist = Int32.(markers)

# The switches ask for a quality mesh (`q`) of the polygon (`p`) with triangles no larger than 3 km² (`a`),
# quietly (`Q`).

triout, _ = triangulate("pQq30a3000000", triin)

x, y = triout.pointlist[1, :], triout.pointlist[2, :]
elements = Int.(triout.trianglelist)
segments, segment_markers = Int.(triout.segmentlist), triout.segmentmarkerlist

# ## The mesh
#
# Both velocity components vanish on the coast and on the grounding line. The front segments carry the
# difference between the ice pressure and the ocean pressure, which pushes the shelf seaward.

coast = falses(length(x))
coast[vec(segments[:, segment_markers .== 1])] .= true
front_segments = segments[:, segment_markers .== 2]

mesh = TriangleMesh(CPU(); x, y, elements, constrained = (coast, coast), front_segments)

# ## Geometry and rheology
#
# The shelf thins from 600 m at the grounding line to 300 m at the front, and floats everywhere above a
# 2 km deep sea floor. Glen's flow law with a uniform rate factor makes it isothermal.

geometry = ice_sheet_geometry(mesh)
auxiliary = ice_sheet_auxiliary_fields(mesh)
set!(geometry.H, (x, y) -> 600 - 300 * x / L)
set!(geometry.b, -2000)

rheology = GlenFlowLaw(mesh; rate_factor = 2.4e-24, exponent = 3)
constants = IceSheetConstants(mesh)

# ## Solving the stress balance
#
# `StressBalanceSolver(mesh)` picks the triangle operator; Picard iterations update the viscosity until the
# velocity changes by less than one part in 10⁸. A floating shelf has no basal friction, so `friction = nothing`.

solver = StressBalanceSolver(mesh; picard_iteration_control = Convergence(Float64; reltol = 1e-8, maxiter = 200))
velocity = VelocityVector(mesh)
solve_stress_balance!(velocity, solver, geometry, auxiliary, rheology, nothing, constants)

u = Array(interior(velocity.u)) .* year
v = Array(interior(velocity.v)) .* year
speed = sqrt.(u .^ 2 .+ v .^ 2)

@info @sprintf("%d nodes, %d elements, %d Picard iterations, maximum speed %.0f m/a",
               length(x), size(elements, 2), solver.iteration, maximum(speed))

# ## Plotting
#
# Makie's `mesh!` colours each triangle by interpolating the nodal values, which is exactly the P1 solution.

points = [Point2f(x[n] / kilometer, y[n] / kilometer) for n in eachindex(x)]

fig = Figure(size = (1000, 450))

ax_mesh = Axis(fig[1, 1]; title = "Mesh", xlabel = "x (km)", ylabel = "y (km)", aspect = DataAspect())
edges = [points[elements[k, e]] for e in axes(elements, 2) for k in (1, 2, 2, 3, 3, 1)]
linesegments!(ax_mesh, edges; color = :gray30, linewidth = 0.3)
linesegments!(ax_mesh, points[vec(front_segments)]; color = :dodgerblue, linewidth = 3, label = "calving front")
axislegend(ax_mesh; position = :lt)

ax_speed = Axis(fig[1, 2]; title = "Speed", xlabel = "x (km)", aspect = DataAspect())
plt = mesh!(ax_speed, points, permutedims(elements); color = speed, colormap = :viridis, shading = NoShading)
Colorbar(fig[1, 3], plt; label = "speed (m/a)")

save("triangle_ice_shelf.png", fig)
