# Conjugate-gradient iterations and solve time of the SSA velocity solve with the Jacobi and multigrid preconditioners
# as the grid is refined. The geometry is a MISMIP+-like marine ice stream (Asay-Davis et al., 2016, Geosci. Model Dev. 9)
# with a prescribed thickness, a grounding line near x = 500 km and a calving front at x = 600 km. The coefficients
# come from a few Picard iterations; each preconditioner then solves the same linear system from zero.
using Oceananigans, Oceananigans.Units, Nunatak, Printf, DelimitedFiles
using Nunatak: ssa_operator!

year = years = 31556926seconds

Lx, Ly = 640kilometers, 80kilometers
B₀, B₂, B₄, B₆, x̄ = -150.0, -728.8, 343.91, -50.57, 300kilometers
fᶜ, dᶜ, wᶜ, z_deep = 4kilometers, 500.0, 24kilometers, -720.0
along_flow_bed(x) = B₀ + B₂ * (x / x̄)^2 + B₄ * (x / x̄)^4 + B₆ * (x / x̄)^6
across_flow_bed(y) = dᶜ / (1 + exp(-2 * (y - Ly / 2 - wᶜ) / fᶜ)) + dᶜ / (1 + exp(2 * (y - Ly / 2 + wᶜ) / fᶜ))
scaling_bed(x, y) = max(along_flow_bed(x) + across_flow_bed(y), z_deep)

x_front = 600kilometers
scaling_thickness(x, y) = ifelse(x < x_front, 1500 * sqrt(1 - x / 650kilometers), 0.0)

function scaling_model(arch, Δx)
    Nx, Ny = round(Int, Lx / Δx), round(Int, Ly / Δx)
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, Lx), y = (0, Ly), topology = (Bounded, Bounded, Flat))
    rheology = GlenFlowLaw(grid; rate_factor = 2.0392e-17 / year, strain_rate_regularization = 1e-5 / year)
    friction = RegularizedCoulombFriction(grid; coefficient = 1e4 * year^(1 / 3), coulomb_coefficient = 0.5, exponent = 1 / 3,
                                          velocity_regularization = 1e-5 / year)
    velocity_solver = StressBalanceSolver(grid; preconditioner = StressBalanceMultigridPreconditioner(grid),
                                          linear_iteration_control = Convergence(; reltol = 1e-6, maxiter = 50_000),
                                          picard_iteration_control = FixedIterations(5))
    model = IceSheetModel(grid; rheology, friction, velocity_solver, bed = scaling_bed,
                          stress_balance = DepthIntegrated(side_walls = FreeSlipWalls()),
                          grounding = SubgridGroundedFraction())
    set!(model, H = scaling_thickness)
    return model
end

function timed_solve(model, preconditioner; reltol, maxiter)
    (; H, φᵘ, φᵛ, N, ice_mask) = model.geometry
    (; ησ, η̄H, η̄Hᶠᶠ, F₂, βᵘ, βᵛ, τᵘ, τᵛ) = model.auxiliary
    coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, model.rheology, model.friction, φᵘ, φᵛ, N, ice_mask, model.stress_balance.side_walls)
    solver = VelocityConjugateGradientSolver(ssa_operator!, model.grid; iteration_control = Convergence(; reltol, maxiter), preconditioner)
    b = VelocityVector(τᵘ, τᵛ)
    x = VelocityVector(model.grid)
    solve!(x, solver, b, coefficients...)
    fill!(x, 0)
    elapsed = @elapsed solve!(x, solver, b, coefficients...)
    return x, solver.iteration, elapsed
end

function multigrid_scaling(; arch = CPU(), resolutions = (8, 4, 2, 1, 0.5) .* kilometers, reltol = 1e-8, maxiter = 100_000,
                             output = "multigrid_scaling")
    rows = Any[]
    for Δx in resolutions
        model = scaling_model(arch, Δx)
        multigrid = StressBalanceMultigridPreconditioner(model.grid)
        jacobi_solution, jacobi_iterations, jacobi_time = timed_solve(model, StressBalanceDiagonalPreconditioner(); reltol, maxiter)
        multigrid_solution, multigrid_iterations, multigrid_time = timed_solve(model, multigrid; reltol, maxiter)
        uᴶ, uᴹ = Array(interior(jacobi_solution.u)), Array(interior(multigrid_solution.u))
        difference = maximum(abs, uᴶ .- uᴹ) / maximum(abs, uᴶ)
        Nx, Ny, _ = size(model.grid)
        levels = length(multigrid.levels)
        coarsest = levels == 0 ? (Nx, Ny) : size(multigrid.levels[end].grid)[1:2]
        push!(rows, (Δx, Nx, Ny, levels, coarsest[1], coarsest[2], jacobi_iterations, jacobi_time, multigrid_iterations, multigrid_time, difference))
        @info @sprintf("Δx = %5.2f km, %5d × %4d, %d levels (coarsest %d × %d): Jacobi %6d iterations %8.2f s, multigrid %5d iterations %8.2f s, max |Δu|/|u| = %.1e",
                       Δx / kilometer, Nx, Ny, levels, coarsest..., jacobi_iterations, jacobi_time, multigrid_iterations, multigrid_time, difference)
    end
    header = ["Δx (m)" "Nx" "Ny" "levels" "coarsest Nx" "coarsest Ny" "Jacobi iterations" "Jacobi time (s)" "multigrid iterations" "multigrid time (s)" "max relative difference"]
    writedlm(output * ".csv", vcat(header, reduce(vcat, [collect(r)' for r in rows])), ',')
    return rows
end
