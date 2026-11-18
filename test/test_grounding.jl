using Nunatak: compute_grounded_fractions!, compute_effective_pressure!, compute_basal_drag!, compute_driving_stress!,
               compute_surface_elevation!

function grounded_fractions(grid, constants, H₀, b₀)
    H, b = CenterField(grid), CenterField(grid)
    set!(H, H₀)
    set!(b, b₀)
    fill_halo_regions!((H, b))
    φ, φᵘ, φᵛ = CenterField(grid), XFaceField(grid), YFaceField(grid)
    compute_grounded_fractions!(φ, φᵘ, φᵛ, SubgridGroundedFraction(), H, b, constants)
    return φ, φᵘ, φᵛ
end

@testset "Grounded fraction and effective pressure" begin
    grid = RectilinearGrid(arch; size = (16, 2), x = (0, 16), y = (0, 2), topology = (Bounded, Periodic, Flat))
    constants = IceSheetConstants(grid)
    ρᵢ, ρw, g = constants.ice_density, constants.ocean_density, constants.gravitational_acceleration
    H₀ = 1000

    for x_GL in (3.3, 7.5, 10.9)
        φ, φᵘ, _ = grounded_fractions(grid, constants, H₀, (x, y) -> ρᵢ / ρw * (50 * (x_GL - x) - H₀))
        φ, φᵘ = Array(interior(φ)), Array(interior(φᵘ))
        @test sum(φ[:, 1, 1]) ≈ x_GL
        @test all(0 .<= φ .<= 1)
        i = floor(Int, x_GL + 1 / 2) + 1
        @test φᵘ[i, 1, 1] ≈ x_GL + 1 / 2 - (i - 1) atol = 1e-12
    end

    φ, φᵘ, φᵛ = grounded_fractions(grid, constants, H₀, -800)
    @test all(Array(interior(φ)) .== 1) && all(Array(interior(φᵘ)) .== 1) && all(Array(interior(φᵛ)) .== 1)
    φ, φᵘ, φᵛ = grounded_fractions(grid, constants, H₀, -1000)
    @test all(Array(interior(φ)) .== 0) && all(Array(interior(φᵘ)) .== 0) && all(Array(interior(φᵛ)) .== 0)

    box = RectilinearGrid(arch; size = (8, 8), x = (0, 8), y = (0, 8), topology = (Bounded, Bounded, Flat))
    unit = IceSheetConstants(box; ice_density = 1, ocean_density = 1)
    for (c, cell, expected) in ((9, (5, 5), 1 / 2), (9, (4, 5), 1), (9.3, (5, 5), 1 - 0.7^2 / 2))
        φ, _, _ = grounded_fractions(box, unit, 0, (x, y) -> c - x - y)
        @test Array(interior(φ))[cell..., 1] ≈ expected
    end

    H, b, N = CenterField(grid), CenterField(grid), CenterField(grid)
    set!(H, H₀)
    set!(b, (x, y) -> x < 4 ? 100 : x < 8 ? -500 : -1000)
    compute_effective_pressure!(N, H, b, constants)
    N = Array(interior(N))
    @test all(N[1:4, :, 1] .≈ ρᵢ * g * H₀)
    @test all(N[5:8, :, 1] .≈ ρᵢ * g * H₀ - ρw * g * 500)
    @test all(N[9:16, :, 1] .== 0)

    H, b, N = CenterField(grid), CenterField(grid), CenterField(grid)
    set!(H, H₀)
    set!(b, (x, y) -> ρᵢ / ρw * (50 * (7.5 - x) - H₀))
    fill_halo_regions!((H, b))
    φ, φᵘ, φᵛ = CenterField(grid), XFaceField(grid), YFaceField(grid)
    compute_grounded_fractions!(φ, φᵘ, φᵛ, SubgridGroundedFraction(), H, b, constants)
    compute_effective_pressure!(N, H, b, constants)
    fill_halo_regions!(N)
    u, v = XFaceField(grid), YFaceField(grid)
    set!(u, 100 / 3.15e7)
    fill_halo_regions!((u, v))
    βᵘ, βᵛ = XFaceField(grid), YFaceField(grid)
    friction = RegularizedCoulombFriction(grid; coefficient = 3e4, coulomb_coefficient = 0.5)
    compute_basal_drag!(βᵘ, βᵛ, friction, u, v, φᵘ, φᵛ, N)
    βᵘ = Array(interior(βᵘ))
    @test all(βᵘ[2:8, :, 1] .> 0)
    @test all(βᵘ[10:16, :, 1] .== 0)
end

@testset "Driving stress split at the grounding line" begin
    ρᵢ, ρw, g = 918, 1028, 9.81
    Δ = 1e3
    for (Nx, Ny, x_GL) in ((16, 2, 7.3e3), (2, 16, 7.3e3))
        grid = RectilinearGrid(arch; size = (Nx, Ny), x = (0, Nx * Δ), y = (0, Ny * Δ), topology = (Bounded, Bounded, Flat))
        constants = IceSheetConstants(grid; ice_density = ρᵢ, ocean_density = ρw, gravitational_acceleration = g)
        along(x, y) = Nx > Ny ? x : y
        H, b, s = CenterField(grid), CenterField(grid), CenterField(grid)
        set!(H, (x, y) -> 1000 - 0.02 * along(x, y))
        set!(b, (x, y) -> ρᵢ / ρw * (0.05 * (x_GL - along(x, y)) - 1000 + 0.02 * along(x, y)))
        fill_halo_regions!((H, b))
        compute_surface_elevation!(s, SubgridGroundedFraction(), H, b, constants)
        fill_halo_regions!(s)

        τ⁰, τ = VelocityVector(grid), VelocityVector(grid)
        compute_driving_stress!(τ⁰, H, s, ρᵢ, ρw, g)
        compute_driving_stress!(τ, H, s, ρᵢ, ρw, g, nothing, b)
        τ⁰ = Array(interior(Nx > Ny ? τ⁰.u : τ⁰.v))
        τ = Array(interior(Nx > Ny ? τ.u : τ.v))
        face(A, n) = Nx > Ny ? A[n, 1, 1] : A[1, n, 1]

        # The grounding line lies in the face between centers 7 (6.5 km) and 8 (7.5 km).
        n = 8
        H₁, H₂ = 1000 - 0.02 * 6.5e3, 1000 - 0.02 * 7.5e3
        f₁, f₂ = 0.05 * (x_GL - 6.5e3), 0.05 * (x_GL - 7.5e3)
        θ = f₁ / (f₁ - f₂)
        Hᵍ = H₁ + θ * (H₂ - H₁)
        s₁, s₂, sᵍ = face(Array(interior(s)), n - 1), (1 - ρᵢ / ρw) * H₂, (1 - ρᵢ / ρw) * Hᵍ
        @test face(τ, n) ≈ - Δ * ρᵢ * g * ((H₁ + Hᵍ) * (sᵍ - s₁) + (Hᵍ + H₂) * (s₂ - sᵍ)) / 2
        @test !(face(τ, n) ≈ face(τ⁰, n))
        others = [m for m in 2:(Nx > Ny ? Nx : Ny) if m != n]
        @test [face(τ, m) for m in others] == [face(τ⁰, m) for m in others]
    end

    grid = RectilinearGrid(arch; size = (16, 4), halo = (4, 4), x = (0, 16e3), y = (0, 4e3), topology = (Bounded, Bounded, Flat))
    τᵘ = map((false, true)) do driving_stress
        model = IceSheetModel(grid; velocity_solver = StressBalanceSolver(grid), rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                              friction = WeertmanFriction(grid; coefficient = 1e6),
                              grounding = SubgridGroundedFraction(; driving_stress), bed = (x, y) -> -600 - 0.05 * x)
        set!(model; H = (x, y) -> 1000 - 0.02 * x)
        Array(interior(model.auxiliary.τᵘ, :, 2, 1))
    end
    straddles = τᵘ[1] .!= τᵘ[2]
    @test findall(straddles) == [5]
    @test summary(SubgridGroundedFraction(driving_stress = true)) == "SubgridGroundedFraction(driving_stress = true)"
end
