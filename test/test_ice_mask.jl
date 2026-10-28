using Nunatak: ice_mask_value

# Ice dome on a flat bed, thinning to `H_edge` outside radius R.
function ice_cap_model(; L = 200e3, Nx = 20, R = 60e3, H₀ = 1000, H_edge = 0, masking_kw = (;),
                       linear_iteration_control = Convergence(; reltol = 1e-10, maxiter = 10_000),
                       picard_iteration_control = Convergence(; reltol = 1e-8, maxiter = 200))
    grid = RectilinearGrid(arch; size = (Nx, Nx), x = (0, L), y = (0, L), topology = (Bounded, Bounded, Flat))
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control, picard_iteration_control)
    model = IceSheetModel(grid; velocity_solver, rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                          friction = WeertmanFriction(grid; coefficient = 3e6),
                          masking = SmoothIceMask(grid; masking_kw...))
    dome(x, y) = (r = hypot(x - L / 2, y - L / 2); r < R ? H₀ * sqrt(1 - (r / R)^2) + H_edge : H_edge)
    set!(model; H = dome)
    return model
end

# Faces with ice-free cells on both sides.
function ice_free_faces(μ; y_walls = true)
    μ = μ[:, :, 1]
    Nx, Ny = size(μ)
    u_free = [i == 1 || i == Nx + 1 || (μ[i-1, j] == 0 && μ[i, j] == 0) for i in 1:Nx+1, j in 1:Ny]
    v_free = y_walls ? [j == 1 || j == Ny + 1 || (μ[i, j-1] == 0 && μ[i, j] == 0) for i in 1:Nx, j in 1:Ny+1] :
                       [μ[i, mod1(j-1, Ny)] == 0 && μ[i, j] == 0 for i in 1:Nx, j in 1:Ny]
    return u_free, v_free
end

@testset "Smooth ice mask" begin
    grid = RectilinearGrid(arch; size = (4, 4), x = (0, 1), y = (0, 1), topology = (Bounded, Bounded, Flat))
    masking = SmoothIceMask(grid; minimum_thickness = 10, transition_thickness = 20)
    μ(H) = ice_mask_value(H, masking)
    @test μ(10.0) == 0 && μ(30.0) == 1 && μ(20.0) == 0.5
    # The slope vanishes at both ends of the transition, so μ is continuously differentiable.
    h = 1e-6
    @test abs(μ(10 + h) - μ(10.0)) / h < 1e-6
    @test abs(μ(30.0) - μ(30 - h)) / h < 1e-6
    @test_throws ArgumentError SmoothIceMask(grid; transition_thickness = 0)

    @testset "Ice-free velocity points are decoupled and the operator stays symmetric" begin
        Nx, Ny = 8, 6
        grid = RectilinearGrid(arch; size = (Nx, Ny), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat))
        ice(x, y) = ifelse(x < 0.5, one(x), zero(x))
        η̄H, η̄Hᶠᶠ, βᵘ, βᵛ = ssa_coefficients(grid; η̄H = (x, y) -> ice(x, y) * (1 + x^2))
        coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ)

        p, y = VelocityVector(grid), VelocityVector(grid)
        set!(p.u, (x, y) -> sin(π * x) + y)
        set!(p.v, (x, y) -> x * y^2 + 1)
        set!(y.u, (x, y) -> cos(x) * y + 1)
        set!(y.v, (x, y) -> sin(3y) + x)
        Ap = apply_ssa_operator(p, coefficients)
        Ay = apply_ssa_operator(y, coefficients)
        @test dot(Ap, y) ≈ dot(p, Ay)
        @test dot(Ap, p) > 0

        u_free, v_free = ice_free_faces(Array(interior(η̄H)); y_walls = false)
        @test count(u_free) > 2Ny && count(v_free) > 0
        # The operator writes the interior faces only, so the east wall is left out.
        u_free[Nx+1, :] .= false
        @test Array(interior(Ap.u))[u_free] == Array(interior(p.u))[u_free]
        @test Array(interior(Ap.v))[v_free] == Array(interior(p.v))[v_free]
    end

    @testset "Ice cap with ice-free margins" begin
        for H_edge in (0, 0.5)
            model = ice_cap_model(; H_edge)
            μ = Array(interior(model.geometry.ice_mask))
            @test count(iszero, μ) > 0
            @test 0 < model.velocity_solver.iteration < 200
            @test all(iszero, Array(interior(model.auxiliary.η̄H))[μ .== 0])

            u, v = Array(interior(model.velocities.u, :, :, 1)), Array(interior(model.velocities.v, :, :, 1))
            u_free, v_free = ice_free_faces(μ)
            @test all(isfinite, u) && all(isfinite, v)
            @test all(iszero, u[u_free]) && all(iszero, v[v_free])
            @test maximum(abs, u) > 0
            # The dome spreads radially, so the velocity is antisymmetric about the centre line.
            @test u ≈ -reverse(u, dims = 1) rtol = 1e-6
        end
    end
end
