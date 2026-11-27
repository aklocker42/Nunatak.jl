using Nunatak: advance_front!
using Oceananigans.Operators: Azᶜᶜᶜ

# Floating channel of full cells up to `Nfull`, with open ocean beyond, under prescribed velocities.
function calving_channel_model(calving; Nx = 12, Ny = 4, Δx = 1e3, H₀ = 100, Nfull = 4, bed = -500,
                               y_topology = Periodic, u = 0, v = 0, rheology = grid -> nothing, grounding = nothing)
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, Nx * Δx), y = (0, Ny * Δx),
                           topology = (Bounded, y_topology, Flat))
    rheology = rheology(grid)
    model = IceSheetModel(grid; velocity_solver = nothing, bed, rheology, grounding, masking = PartialCellFront(grid; calving = calving(grid)))
    set!(model; H = (x, y) -> ifelse(x < Nfull * Δx, H₀, 0), u, v)
    return model
end

calved_volume(model) = sum(Array(interior(model.masking.calved_thickness))) * Azᶜᶜᶜ(1, 1, 1, model.grid)
ice_volume(model) = sum(Array(interior(model.geometry.H))) * Azᶜᶜᶜ(1, 1, 1, model.grid)
cpu_field(f) = Array(interior(f, :, :, 1))

@testset "Calving" begin
    @testset "Thickness threshold removes the marine front column and closes the budget" begin
        H₀, Nfull = 100, 4
        model = calving_channel_model(grid -> ThicknessThresholdCalving(grid; thickness = 2H₀); H₀, Nfull)
        V₀ = ice_volume(model)
        advance_front!(model.masking, model, 1)
        H, μ = cpu_field(model.geometry.H), cpu_field(model.geometry.ice_mask)
        @test all(H[1:Nfull - 1, :] .== H₀)
        @test all(μ[1:Nfull - 1, :] .== 1)
        @test all(iszero, H[Nfull:end, :])
        @test all(iszero, μ[Nfull:end, :])
        @test all(cpu_field(model.masking.calved_thickness)[Nfull, :] .== H₀)
        @test ice_volume(model) + calved_volume(model) ≈ V₀
    end

    @testset "Ice thicker than the threshold and land-terminating fronts do not calve" begin
        for (thickness, bed) in ((50, -500), (200, 100))
            model = calving_channel_model(grid -> ThicknessThresholdCalving(grid; thickness); bed)
            H₀ = cpu_field(model.geometry.H)
            advance_front!(model.masking, model, 1)
            @test cpu_field(model.geometry.H) == H₀
            @test all(iszero, cpu_field(model.masking.calved_thickness))
        end
    end

    @testset "A thin partial cell calves and leaves the full cells behind it" begin
        H₀, Nfull = 100, 4
        model = calving_channel_model(grid -> ThicknessThresholdCalving(grid; thickness = 2H₀); H₀, Nfull)
        H, μ, Hʳ = model.geometry.H, model.geometry.ice_mask, model.geometry.front_thickness
        interior(H, Nfull + 1, :, 1) .= H₀ / 2
        interior(Hʳ, Nfull + 1, :, 1) .= H₀
        fill_halo_regions!((H, μ, Hʳ))
        advance_front!(model.masking, model, 1)
        @test all(iszero, cpu_field(H)[Nfull + 1, :])
        @test all(cpu_field(H)[1:Nfull, :] .== H₀)
        @test all(cpu_field(μ)[1:Nfull, :] .== 1)
        @test all(cpu_field(model.masking.calved_thickness)[Nfull + 1, :] .≈ H₀ / 2)
    end

    @testset "Time stepping calves one column per step until the ice is gone" begin
        H₀, Nfull = 100, 4
        model = calving_channel_model(grid -> ThicknessThresholdCalving(grid; thickness = 2H₀); H₀, Nfull)
        V₀ = ice_volume(model)
        for n in 1:Nfull
            time_step!(model, 1)
            @test all(cpu_field(model.geometry.ice_mask)[1:Nfull - n, :] .== 1)
            @test all(iszero, cpu_field(model.geometry.H)[Nfull - n + 1:end, :])
        end
        @test calved_volume(model) ≈ V₀
    end

    @testset "Eigencalving retreat of a full front cell under biaxial spreading" begin
        H₀, Nfull, Δx, Ny, a, b, K, Δt = 100, 4, 1e3, 6, 1e-9, 1e-9, 1e13, 1e7
        model = calving_channel_model(grid -> EigenCalving(grid; coefficient = K); H₀, Nfull, Δx, Ny,
                                      y_topology = Bounded, u = (x, y) -> a * x, v = (x, y) -> b * y,
                                      rheology = grid -> GlenFlowLaw(grid; rate_factor = 1e-24))
        V₀ = ice_volume(model)
        advance_front!(model.masking, model, Δt)
        ε̇₀ = model.rheology.strain_rate_regularization
        # Beyond the front `v` is zero, so the shear at the front cell is `-b y / 4Δx`; it closes the second
        # principal direction from the fifth row on, and the north wall stops the spreading in the last.
        rows = 1:Ny - 2
        f = map(rows) do j
            y = (j - 1 / 2) * Δx
            ε̇xy = -b * y / 4Δx
            r = sqrt(((a - b) / 2)^2 + ε̇xy^2 + ε̇₀^2)
            c = K * max((a + b) / 2 + r, 0) * max((a + b) / 2 - r, 0)
            # The east face is projected across the flow.
            ū, v̄ = a * (Nfull - 1 / 2) * Δx, b * y
            return min(1, c * Δt * ū / sqrt(ū^2 + v̄^2) / Δx)
        end
        H = cpu_field(model.geometry.H)
        @test all(0 .< f .< 1)
        @test H[Nfull, rows] ≈ H₀ .* (1 .- f)
        @test H[Nfull, Ny - 1:Ny] == [H₀, H₀]
        @test all(cpu_field(model.geometry.ice_mask)[Nfull, rows] .== 0)
        @test all(cpu_field(model.geometry.front_thickness)[Nfull, rows] .≈ H₀)
        @test all(H[1:Nfull - 1, :] .== H₀)
        @test ice_volume(model) + calved_volume(model) ≈ V₀
    end

    @testset "Von Mises retreat of a full front cell and of the partial cell it becomes" begin
        H₀, Nfull, Δx, a, η̄, σᵐᵃˣ, Δt = 100, 4, 1e3, 1e-9, 1e14, 3e5, 1e7
        model = calving_channel_model(grid -> VonMisesCalving(grid; stress_threshold = σᵐᵃˣ); H₀, Nfull, Δx, u = (x, y) -> a * x,
                                      rheology = grid -> GlenFlowLaw(grid; rate_factor = 1e-24))
        set!(model.auxiliary.η̄H, (x, y) -> ifelse(x < Nfull * Δx, η̄ * H₀, 0))
        fill_halo_regions!(model.auxiliary.η̄H)
        n, ε̇₀ = model.rheology.exponent, model.rheology.strain_rate_regularization
        ε̇₁ = a / 2 + sqrt(a^2 / 4 + ε̇₀^2)
        ε̃² = ε̇₁^2 / 2
        B = 2η̄ * (a^2 + ε̇₀^2)^((n - 1) / 2n)
        σ̃ = √3 * B * ε̃² * (ε̃² + ε̇₀^2)^((1 - 2n) / 2n)
        speed(ū) = sqrt(ū^2 + (ε̇₀ * Δx)^2)
        f = speed(Nfull * a * Δx) * σ̃ / σᵐᵃˣ * Δt / Δx
        V₀ = ice_volume(model)
        advance_front!(model.masking, model, Δt)
        @test 0 < f < 1
        @test all(cpu_field(model.geometry.H)[Nfull, :] .≈ H₀ * (1 - f))
        @test ice_volume(model) + calved_volume(model) ≈ V₀

        # The partial cell takes its rate from its full upstream neighbor, at that neighbor's faster face.
        f′ = speed((Nfull - 1) * a * Δx) * σ̃ / σᵐᵃˣ * Δt / Δx
        advance_front!(model.masking, model, Δt)
        @test all(cpu_field(model.geometry.H)[Nfull, :] .≈ H₀ * (1 - f - f′))
        @test ice_volume(model) + calved_volume(model) ≈ V₀
    end

    @testset "Front velocity calving moves a floating front at the set velocity" begin
        H₀, Nfull, Δx, U, Δt = 100, 4, 1e3, 1e-6, 1e8
        for w in (0, -2U)
            model = calving_channel_model(grid -> FrontVelocityCalving(grid; front_velocity = t -> w); H₀, Nfull, Δx, u = U,
                                          grounding = SubgridGroundedFraction())
            V₀ = ice_volume(model)
            advance_front!(model.masking, model, Δt)
            f = (U - w) * Δt / Δx
            @test all(cpu_field(model.geometry.H)[Nfull, :] .≈ H₀ * (1 - f))
            @test all(cpu_field(model.geometry.ice_mask)[Nfull, :] .== 0)
            @test all(cpu_field(model.geometry.H)[1:Nfull - 1, :] .== H₀)
            @test ice_volume(model) + calved_volume(model) ≈ V₀
        end
    end

    @testset "Front velocity calving passes the retreat an emptied partial cell could not take to the ice behind" begin
        H₀, Nfull, Δx, U, Δt, w = 100, 4, 1e3, 1e-6, 1e8, -5e-6
        model = calving_channel_model(grid -> FrontVelocityCalving(grid; front_velocity = w); H₀, Nfull, Δx, u = U,
                                      grounding = SubgridGroundedFraction())
        H, μ, Hʳ = model.geometry.H, model.geometry.ice_mask, model.geometry.front_thickness
        interior(H, Nfull + 1, :, 1) .= H₀ / 4
        interior(Hʳ, Nfull + 1, :, 1) .= H₀
        fill_halo_regions!((H, μ, Hʳ))
        V₀ = ice_volume(model)
        advance_front!(model.masking, model, Δt)
        f = (U - w) * Δt / Δx
        @test 1 / 4 < f < 1
        @test all(iszero, cpu_field(H)[Nfull + 1, :])
        @test all(cpu_field(H)[Nfull, :] .≈ H₀ * (1 - (f - 1 / 4)))
        @test all(cpu_field(μ)[Nfull, :] .== 0)
        @test all(cpu_field(H)[1:Nfull - 1, :] .== H₀)
        @test ice_volume(model) + calved_volume(model) ≈ V₀
    end

    @testset "Maximum extent calving removes the front column outside the extent and keeps the rest" begin
        H₀, Nfull, Δx = 100, 4, 1e3
        model = calving_channel_model(grid -> MaximumExtentCalving(grid; extent = (x, y) -> (Nfull - 1) * Δx - x); H₀, Nfull, Δx)
        V₀ = ice_volume(model)
        for _ in 1:2
            advance_front!(model.masking, model, 1)
            @test all(iszero, cpu_field(model.geometry.H)[Nfull:end, :])
            @test all(cpu_field(model.geometry.H)[1:Nfull - 1, :] .== H₀)
            @test all(cpu_field(model.geometry.ice_mask)[1:Nfull - 1, :] .== 1)
        end
        @test ice_volume(model) + calved_volume(model) ≈ V₀
    end
end
