using Nunatak: compute_driving_stress!

@testset "Calving front" begin
    @testset "Front force of floating, marine and land cliffs" begin
        Nx, Ny = 6, 4
        grid = RectilinearGrid(arch; size = (Nx, Ny), x = (0, 6e3), y = (0, 4e3), topology = (Bounded, Bounded, Flat))
        ρᵢ, ρw, g, H₀ = 918, 1028, 9.81, 500
        for (b, D) in ((-2000, ρᵢ * H₀ / ρw), (-300, 300), (100, 0))
            H, s, μ = CenterField(grid), CenterField(grid), CenterField(grid)
            set!(H, (x, y) -> ifelse(x < 3e3, H₀, 0))
            set!(μ, (x, y) -> ifelse(x < 3e3, 1, 0))
            # Surface of floating or grounded ice; zero where the open ocean or land is bare.
            set!(s, (x, y) -> ifelse(x < 3e3, max(b + H₀, H₀ * (1 - ρᵢ / ρw)), 0))
            fill_halo_regions!((H, s, μ))
            τ = VelocityVector(grid)
            compute_driving_stress!(τ, H, s, ρᵢ, ρw, g, μ)
            τu = Array(interior(τ.u))
            @test τu[4, :, 1] ≈ fill(1e3 * g * (ρᵢ * H₀^2 - ρw * D^2) / 2, Ny)
            @test all(iszero, τu[5:end, :, 1])
        end
    end

    @testset "Confined floating shelf matches the analytic spreading rate" begin
        L, x_front, H₀, A, Nx = 64e3, 40e3, 500, 1e-24, 32
        grid = RectilinearGrid(arch; size = (Nx, 4), halo = (4, 4), x = (0, L), y = (0, 4L / Nx), topology = (Bounded, Bounded, Flat))
        velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000),
                                              picard_iteration_control = Convergence(; reltol = 1e-10, maxiter = 500))
        model = IceSheetModel(grid; velocity_solver, rheology = GlenFlowLaw(grid; rate_factor = A),
                              friction = WeertmanFriction(grid; coefficient = 1e6),
                              grounding = SubgridGroundedFraction(), bed = -2000)
        set!(model; H = (x, y) -> ifelse(x < x_front, H₀, 0), u = (x, y) -> 1e-9 * x)
        c = model.constants
        ρ′ = c.ice_density * (1 - c.ice_density / c.ocean_density)
        ε̇ = A * (ρ′ * c.gravitational_acceleration * H₀ / 4)^3
        x = xnodes(grid, Face())
        u = Array(interior(model.velocities.u, :, 2, 1))
        ice = x .<= x_front
        @test maximum(abs, u[ice] .- ε̇ .* x[ice]) / (ε̇ * x_front) < 1e-6
        @test maximum(abs, Array(interior(model.velocities.v))) < 1e-12 * ε̇ * x_front
    end

    @testset "Steady shelf with constant flux converges at second order" begin
        # With uH = q and a floating front, ∂u/∂x = A (ρ′ g H / 4)ⁿ fixes H(x); the west wall gives u = q/H - q/H₀.
        L, x_front, H₀, A, n = 256e3, 192e3, 1e3, 1e-25, 3
        q = H₀ * 1e-5
        errors = map((64, 128)) do Nx
            grid = RectilinearGrid(arch; size = (Nx, 4), halo = (4, 4), x = (0, L), y = (0, 4L / Nx), topology = (Bounded, Bounded, Flat))
            velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 20_000),
                                                  picard_iteration_control = Convergence(; reltol = 1e-11, maxiter = 500))
            model = IceSheetModel(grid; velocity_solver, rheology = GlenFlowLaw(grid; rate_factor = A),
                                  friction = WeertmanFriction(grid; coefficient = 1e6),
                                  grounding = SubgridGroundedFraction(), bed = -2000)
            c = model.constants
            ρ′ = c.ice_density * (1 - c.ice_density / c.ocean_density)
            k = (n + 1) * (A / q) * (ρ′ * c.gravitational_acceleration / 4)^n
            H(x) = (H₀^-(n + 1) + k * x)^(-1 / (n + 1))
            set!(model; H = (x, y) -> ifelse(x < x_front, H(x), 0), u = (x, y) -> 1e-9 * x)
            x = xnodes(grid, Face())
            ice = x .<= x_front
            u = Array(interior(model.velocities.u, :, 2, 1))[ice]
            u_exact = q ./ H.(x[ice]) .- q / H₀
            return maximum(abs, u .- u_exact) / maximum(u_exact)
        end
        @test errors[2] < 5e-3
        @test errors[1] / errors[2] > 3
    end
end
