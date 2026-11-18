# Confined square ice shelf on P1 triangles (ISSM/DJUICE test101):
# compare the shallow-shelf velocities with DJUICE on the same meshes at 50, 20, 10 and 5 km.
using Oceananigans, Nunatak, DelimitedFiles, Printf
using Nunatak: ice_sheet_geometry, ice_sheet_auxiliary_fields

yts = 365 * 24 * 3600
ρᵢ, ρw, g = 917.0, 1023.0, 9.81
B, n = 1.815730284801701e8, 3
Hmin, Hmax = 300.0, 1000.0

djuice_reference_directory = joinpath(@__DIR__, "square_shelf_djuice_reference")

function square_shelf_triangle(label; arch = CPU(), reference_directory = djuice_reference_directory)
    nodes = readdlm(joinpath(reference_directory, "nodes_$label.csv"), ',', Float64)
    elements = permutedims(readdlm(joinpath(reference_directory, "elements_$label.csv"), ',', Int))
    x, y = nodes[:, 1], nodes[:, 2]
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)

    mesh = TriangleMesh(arch; x, y, elements)
    geometry, auxiliary = ice_sheet_geometry(mesh), ice_sheet_auxiliary_fields(mesh)
    set!(geometry.H, (x, y) -> Hmax + (Hmin - Hmax) * ((y - ymin) / (ymax - ymin) + (x - xmin) / (xmax - xmin) / 10))
    set!(geometry.b, (x, y) -> -ρᵢ / ρw * (Hmax + (Hmin - Hmax) * ((y - ymin) / (ymax - ymin) + (x - xmin) / (xmax - xmin) / 10)) - 10)

    rheology = GlenFlowLaw(mesh; rate_factor = B^(-n), exponent = n)
    constants = IceSheetConstants(mesh; ice_density = ρᵢ, ocean_density = ρw, gravitational_acceleration = g)
    solver = StressBalanceSolver(mesh; picard_iteration_control = Convergence(Float64; reltol = 1e-8, maxiter = 200))
    velocity = VelocityVector(mesh)

    solve_stress_balance!(velocity, solver, geometry, auxiliary, rheology, nothing, constants)
    fill!(velocity, 0)
    elapsed = @elapsed solve_stress_balance!(velocity, solver, geometry, auxiliary, rheology, nothing, constants)

    u = Array(interior(velocity.u)) .* yts
    v = Array(interior(velocity.v)) .* yts
    uᵣ, vᵣ = nodes[:, 6], nodes[:, 7]
    error = sqrt(sum((u .- uᵣ) .^ 2 .+ (v .- vᵣ) .^ 2) / sum(uᵣ .^ 2 .+ vᵣ .^ 2))
    speed = maximum(sqrt.(u .^ 2 .+ v .^ 2))

    @printf("%s  nodes %6d  elements %6d  max speed %8.2f m/a  relative L2 error %.3e  Picard %3d  last CG %5d  solve %.2f s\n",
            label, length(x), size(elements, 2), speed, error, solver.iteration, solver.linear_solver.iteration, elapsed)
    return (; label, Nnodes = length(x), Nelements = size(elements, 2), speed, error,
              picard = solver.iteration, cg = solver.linear_solver.iteration, elapsed)
end

# The 5 km mesh is not in the repository; copy nodes_05km.csv and elements_05km.csv into djuice_reference_directory to include it.
labels = filter(label -> isfile(joinpath(djuice_reference_directory, "nodes_$label.csv")), ("50km", "20km", "10km", "05km"))
results = [square_shelf_triangle(label) for label in labels]
