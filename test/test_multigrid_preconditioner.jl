using Nunatak: ssa_operator!, restrict!, prolongate!, coarsen_coefficients!, _interpolate_to_corners!
using Oceananigans.Utils: launch!

solved_values(f) = Array(interior(f))[1:size(f.grid, 1), 1:size(f.grid, 2), 1]
solved_dot(a, b) = sum(solved_values(a.u) .* solved_values(b.u)) + sum(solved_values(a.v) .* solved_values(b.v))

function smooth_vector(grid, m)
    x = VelocityVector(grid)
    set!(x.u, (x, y) -> sin(2π * m * x + 1) * cos(2π * y + m))
    set!(x.v, (x, y) -> cos(2π * x + m) * sin(2π * m * y + 2))
    fill_halo_regions!(x)
    return x
end

# Stiffness with an ice-free hole, so the transfers see inactive velocity points.
function holed_coefficients(grid)
    η̄H = Field{Center, Center, Nothing}(grid)
    η̄Hᶠᶠ = Field{Face, Face, Nothing}(grid)
    βᵘ, βᵛ = Field{Face, Center, Nothing}(grid), Field{Center, Face, Nothing}(grid)
    set!(η̄H, (x, y) -> ifelse((x - 0.7)^2 + (y - 0.4)^2 < 0.04, 0, 1 + x))
    fill_halo_regions!(η̄H)
    launch!(arch, grid, size(η̄Hᶠᶠ), _interpolate_to_corners!, η̄Hᶠᶠ, η̄H, grid)
    fill_halo_regions!(η̄Hᶠᶠ)
    set!(βᵘ, 3)
    set!(βᵛ, 3)
    return η̄H, η̄Hᶠᶠ, βᵘ, βᵛ
end

@testset "Multigrid preconditioner" begin
    grid = RectilinearGrid(arch; size = (32, 16), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat))
    @test map(level -> size(level.grid), StressBalanceMultigridPreconditioner(grid).levels) == ((16, 8, 1), (8, 4, 1))
    odd_grid = RectilinearGrid(arch; size = (15, 8), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat))
    @test (@test_logs (:warn, r"cannot be halved") StressBalanceMultigridPreconditioner(odd_grid)).levels == ()

    for topology in ((Bounded, Periodic, Flat), (Bounded, Bounded, Flat), (Periodic, Periodic, Flat))
        @testset "Transfers and symmetry on $topology" begin
            grid = RectilinearGrid(arch; size = (16, 8), halo = (4, 4), x = (0, 1), y = (0, 1), topology)
            P = StressBalanceMultigridPreconditioner(grid)
            η̄H, η̄Hᶠᶠ, βᵘ, βᵛ = holed_coefficients(grid)
            coarsen_coefficients!(P.levels, η̄H, βᵘ, βᵛ)
            coarse = first(P.levels)

            f, c = smooth_vector(grid, 3), smooth_vector(coarse.grid, 2)
            Rf, Pc = VelocityVector(coarse.grid), VelocityVector(grid)
            restrict!(Rf, f, coarse.η̄H, η̄H)
            prolongate!(Pc, c, η̄H, coarse.η̄H)
            @test solved_dot(Rf, c) ≈ solved_dot(f, Pc) rtol = 1e-12

            r₁, r₂ = smooth_vector(grid, 1), smooth_vector(grid, 2)
            z₁, z₂ = VelocityVector(grid), VelocityVector(grid)
            precondition!(z₁, P, r₁, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, FreeSlipWalls())
            precondition!(z₂, P, r₂, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, FreeSlipWalls())
            @test solved_dot(z₁, r₂) ≈ solved_dot(r₁, z₂) rtol = 1e-10
            @test solved_dot(z₁, r₁) > 0
        end
    end

    @testset "Conjugate gradients on an ice cap" begin
        model = ice_cap_model(; Nx = 128, masking_kw = (; minimum_thickness = 10, transition_thickness = 400))
        (; η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, τᵘ, τᵛ) = model.auxiliary
        b = VelocityVector(τᵘ, τᵛ)
        control = Convergence(; reltol = 1e-10, maxiter = 20_000)
        solutions = map((StressBalanceDiagonalPreconditioner(), StressBalanceMultigridPreconditioner(model.grid))) do preconditioner
            solver = VelocityConjugateGradientSolver(ssa_operator!, model.grid; iteration_control = control, preconditioner)
            x = VelocityVector(model.grid)
            solve!(x, solver, b, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, FreeSlipWalls())
            return x, solver.iteration
        end
        (jacobi, jacobi_iterations), (multigrid, multigrid_iterations) = solutions
        @info "Ice cap CG iterations" jacobi_iterations multigrid_iterations
        @test multigrid_iterations < jacobi_iterations / 3
        @test solved_values(multigrid.u) ≈ solved_values(jacobi.u) rtol = 1e-7
        @test solved_values(multigrid.v) ≈ solved_values(jacobi.v) rtol = 1e-7
    end
end
