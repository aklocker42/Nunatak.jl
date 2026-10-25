using Nunatak: compute_grounded_fractions!, compute_effective_pressure!, compute_basal_drag!

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
