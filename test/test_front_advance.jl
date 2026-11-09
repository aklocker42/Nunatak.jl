using Nunatak: advance_front!
using Oceananigans.Operators: Azᶜᶜᶜ

# Channel of full cells with thickness H₀ up to `Nfull`, moved by a prescribed uniform velocity U.
function front_channel_model(; Nx = 16, Ny = 4, Δx = 1e3, U = 1, H₀ = 100, Nfull = 4, y_topology = Periodic,
                             advection = UpwindBiased(order = 1), surface_mass_balance = 0, passes = 4)
    L = Nx * Δx
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, L), y = (0, Ny * Δx), topology = (Bounded, y_topology, Flat))
    model = IceSheetModel(grid; velocity_solver = nothing, advection, surface_mass_balance,
                          masking = PartialCellFront(grid; passes))
    set!(model; H = (x, y) -> ifelse(x < Nfull * Δx, H₀, 0), u = (x, y) -> ifelse(0 < x < L, U, 0))
    return model
end

total_volume(model) = sum(Array(interior(model.geometry.H))) * Azᶜᶜᶜ(1, 1, 1, model.grid)
cpu_interior(f) = Array(interior(f, :, :, 1))
Δx_area(model) = Azᶜᶜᶜ(1, 1, 1, model.grid)

@testset "Partial-cell front advance" begin
    @testset "Front advances one cell per Δx / U under a steady inflow" begin
        Δx, U, H₀, Nfull = 1e3, 1, 100, 4
        model = front_channel_model(; Δx, U, H₀, Nfull, surface_mass_balance = (x, y) -> ifelse(x < Δx, U * H₀ / Δx, 0))
        Δt = Δx / U / 4
        μ = cpu_interior(model.geometry.ice_mask)
        @test count(μ[:, 1] .== 1) == Nfull
        for n in 1:20
            time_step!(model, Δt)
        end
        H = cpu_interior(model.geometry.H)
        μ = cpu_interior(model.geometry.ice_mask)
        @test all(μ[1:Nfull + 5, :] .== 1)
        @test all(μ[Nfull + 6:end, :] .== 0)
        @test all(H[1:Nfull + 5, :] .== H₀)
        @test all(iszero, H[Nfull + 6:end, :])
    end

    @testset "Overfilled front cell spills into empty neighbours and conserves volume" begin
        H₀, Nfull = 100, 4
        model = front_channel_model(; H₀, Nfull)
        H, μ = model.geometry.H, model.geometry.ice_mask
        interior(H, Nfull + 1, :, 1) .= 2.5H₀
        interior(μ, Nfull + 1, :, 1) .= 0
        fill_halo_regions!((H, μ))
        V₀ = total_volume(model)
        advance_front!(model.masking, model, 1)
        H = cpu_interior(H)
        μ = cpu_interior(μ)
        @test all(H[1:Nfull + 2, :] .≈ H₀)
        @test all(H[Nfull + 3, :] .≈ H₀ / 2)
        @test all(μ[1:Nfull + 2, :] .== 1)
        @test all(μ[Nfull + 3:end, :] .== 0)
        @test total_volume(model) ≈ V₀
    end

    @testset "Each pass moves the overflow one cell" begin
        H₀, Nfull = 100, 4
        model = front_channel_model(; H₀, Nfull, passes = 1)
        H, μ = model.geometry.H, model.geometry.ice_mask
        interior(H, Nfull + 1, :, 1) .= 3H₀
        interior(μ, Nfull + 1, :, 1) .= 0
        fill_halo_regions!((H, μ))
        advance_front!(model.masking, model, 1)
        H = cpu_interior(H)
        @test all(H[Nfull + 1, :] .≈ H₀)
        @test all(H[Nfull + 2, :] .≈ 2H₀)
        @test all(cpu_interior(μ)[Nfull + 2, :] .== 0)
    end

    @testset "A front cell at a wall thickens instead of spilling into the halo" begin
        Nx, H₀ = 8, 100
        model = front_channel_model(; Nx, H₀, Nfull = Nx, y_topology = Bounded)
        H, μ = model.geometry.H, model.geometry.ice_mask
        interior(H, Nx, :, 1) .= 2H₀
        interior(μ, Nx, :, 1) .= 0
        fill_halo_regions!((H, μ))
        V₀ = total_volume(model)
        advance_front!(model.masking, model, 1)
        @test all(cpu_interior(H)[Nx, :] .≈ 2H₀)
        @test all(cpu_interior(μ) .== 1)
        @test total_volume(model) ≈ V₀
    end

    @testset "Drained full cells become empty" begin
        model = front_channel_model(; Nfull = 4)
        interior(model.geometry.H, 1, :, 1) .= 0
        advance_front!(model.masking, model, 1)
        μ = cpu_interior(model.geometry.ice_mask)
        @test all(μ[1, :] .== 0)
        @test all(μ[2:4, :] .== 1)
    end

    @testset "Surface mass balance thickens a partial cell without changing its covered fraction" begin
        H₀, Nfull, a, Δt = 100, 4, 1e-3, 1e3
        model = front_channel_model(; H₀, Nfull, U = 0, surface_mass_balance = a)
        H, μ, Hʳ = model.geometry.H, model.geometry.ice_mask, model.geometry.front_thickness
        interior(H, Nfull + 1, :, 1) .= H₀ / 4
        interior(Hʳ, Nfull + 1, :, 1) .= H₀
        fill_halo_regions!((H, μ, Hʳ))
        V₀ = total_volume(model)
        advance_front!(model.masking, model, Δt)
        @test all(cpu_interior(Hʳ)[Nfull + 1, :] .≈ H₀ + a * Δt)
        @test all(cpu_interior(H)[Nfull + 1, :] .≈ (H₀ + a * Δt) / 4)
        @test all(cpu_interior(H)[1:Nfull, :] .== H₀)
        @test all(cpu_interior(μ)[Nfull + 1, :] .== 0)
        @test total_volume(model) ≈ V₀ + a * Δt / 4 * Δx_area(model) * model.grid.Ny
    end

    @testset "Ablation that removes all the ice empties a partial cell and closes the budget" begin
        H₀, Nfull, Δt = 100, 4, 1e3
        a = -2H₀ / Δt
        model = front_channel_model(; H₀, Nfull, U = 0, surface_mass_balance = (x, y) -> ifelse(x > Nfull * 1e3, a, 0))
        H, μ, Hʳ = model.geometry.H, model.geometry.ice_mask, model.geometry.front_thickness
        interior(H, Nfull + 1, :, 1) .= H₀ / 2
        interior(Hʳ, Nfull + 1, :, 1) .= H₀
        fill_halo_regions!((H, μ, Hʳ))
        V₀ = total_volume(model)
        advance_front!(model.masking, model, Δt)
        @test all(iszero, cpu_interior(H)[Nfull + 1, :])
        @test all(iszero, cpu_interior(Hʳ)[Nfull + 1, :])
        @test all(cpu_interior(μ)[Nfull + 1, :] .== 0)
        clipped = sum(cpu_interior(model.geometry.clipped_thickness)) * Δx_area(model)
        @test total_volume(model) - clipped ≈ V₀ + a * Δt / 2 * Δx_area(model) * model.grid.Ny
    end

    @testset "A cell that fills is forced as full and its spill as partial" begin
        H₀, Nfull, a, Δt = 100, 4, 1e-3, 1e3
        model = front_channel_model(; H₀, Nfull, surface_mass_balance = a)
        H, μ, Hʳ = model.geometry.H, model.geometry.ice_mask, model.geometry.front_thickness
        interior(H, Nfull + 1, :, 1) .= 1.5H₀
        fill_halo_regions!((H, μ))
        V₀ = total_volume(model)
        advance_front!(model.masking, model, Δt)
        Hᶜ = cpu_interior(H)
        @test all(Hᶜ[Nfull + 1, :] .≈ H₀ + a * Δt)
        @test all(Hᶜ[Nfull + 2, :] .≈ (H₀ + a * Δt) / 2)
        @test all(iszero, cpu_interior(Hʳ)[Nfull + 1, :])
        @test all(cpu_interior(Hʳ)[Nfull + 2, :] .≈ H₀ + a * Δt)
        @test total_volume(model) ≈ V₀ + 1.5a * Δt * Δx_area(model) * model.grid.Ny
    end

    @testset "A partial cell with no inflow keeps its ice thickness" begin
        H₀, Nfull = 100, 4
        model = front_channel_model(; H₀, Nfull, U = 0)
        H, Hʳ = model.geometry.H, model.geometry.front_thickness
        interior(H, Nfull + 1, :, 1) .= H₀ / 4
        interior(Hʳ, Nfull + 1, :, 1) .= 3H₀
        fill_halo_regions!((H, Hʳ))
        advance_front!(model.masking, model, 1)
        @test all(cpu_interior(Hʳ)[Nfull + 1, :] .== 3H₀)
        @test all(cpu_interior(H)[Nfull + 1, :] .== H₀ / 4)
    end

    @testset "Ice without inflow or ice thickness takes it from full neighbours, or fills" begin
        H₀, Nfull, a, Δt = 100, 4, 1e-3, 1e3
        model = front_channel_model(; H₀, Nfull, U = 0, surface_mass_balance = a)
        H, μ, Hʳ = model.geometry.H, model.geometry.ice_mask, model.geometry.front_thickness
        interior(H, Nfull + 1, :, 1) .= H₀ / 4
        interior(H, Nfull + 4, :, 1) .= H₀ / 4
        fill_halo_regions!(H)
        advance_front!(model.masking, model, Δt)
        @test all(cpu_interior(Hʳ)[Nfull + 1, :] .≈ H₀ + a * Δt)
        @test all(cpu_interior(H)[Nfull + 1, :] .≈ (H₀ + a * Δt) / 4)
        @test all(cpu_interior(μ)[Nfull + 4, :] .== 1)
        @test all(cpu_interior(H)[Nfull + 4, :] .≈ H₀ / 4 + a * Δt)
    end

    @testset "Overflow across a periodic boundary sees the spilling cell's new thickness" begin
        H₀, Nfull = 100, 4
        model = front_channel_model(; H₀, Nfull)
        H, μ, Hʳ = model.geometry.H, model.geometry.ice_mask, model.geometry.front_thickness
        interior(H, Nfull + 1, :, 1) .= 0
        interior(H, Nfull + 1, 1:1, 1) .= 2.5H₀
        interior(μ, Nfull + 1, :, 1) .= 0
        fill_halo_regions!((H, μ))
        advance_front!(model.masking, model, 1)
        @test cpu_interior(Hʳ)[Nfull + 1, 2] == H₀
        @test cpu_interior(Hʳ)[Nfull + 1, 4] == H₀
    end

    @testset "Bounded WENO transport conserves volume through the moving front" begin
        Δx, U, H₀, Nfull = 1e3, 1, 100, 4
        model = front_channel_model(; Δx, U, H₀, Nfull, advection = WENO(order = 5, bounds = (0.0, 1e5)),
                                    surface_mass_balance = (x, y) -> ifelse(x < Δx, U * H₀ / Δx, 0))
        Δt = Δx / U / 4
        V₀ = total_volume(model)
        Nt = 20
        for n in 1:Nt
            time_step!(model, Δt)
        end
        supplied = Nt * Δt * U * H₀ * model.grid.Ly
        clipped = sum(cpu_interior(model.geometry.clipped_thickness)) * Δx^2
        @test total_volume(model) - clipped ≈ V₀ + supplied rtol = 1e-12
        μ = cpu_interior(model.geometry.ice_mask)
        @test count(μ[:, 1] .== 1) ≥ Nfull + 4
        @test all(cpu_interior(model.geometry.H) .≥ 0)
    end

    @testset "Spreading floating shelf advances its front under the stress balance" begin
        L, x_front, H₀, A, Nx = 64e3, 40e3, 500, 1e-24, 32
        grid = RectilinearGrid(arch; size = (Nx, 4), halo = (4, 4), x = (0, L), y = (0, 4L / Nx), topology = (Bounded, Periodic, Flat))
        velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-10, maxiter = 10_000),
                                              picard_iteration_control = Convergence(; reltol = 1e-8, maxiter = 200))
        model = IceSheetModel(grid; velocity_solver, rheology = GlenFlowLaw(grid; rate_factor = A),
                              friction = WeertmanFriction(grid; coefficient = 1e6),
                              grounding = SubgridGroundedFraction(), bed = -2000, masking = PartialCellFront(grid))
        set!(model; H = (x, y) -> ifelse(x < x_front, H₀, 0), u = (x, y) -> 1e-9 * x)
        V₀ = total_volume(model)
        Nfull₀ = count(cpu_interior(model.geometry.ice_mask)[:, 1] .== 1)
        for n in 1:10
            time_step!(model, 5e6)
        end
        μ = cpu_interior(model.geometry.ice_mask)
        clipped = sum(cpu_interior(model.geometry.clipped_thickness)) * Azᶜᶜᶜ(1, 1, 1, grid)
        @test total_volume(model) - clipped ≈ V₀ rtol = 1e-12
        @test count(μ[:, 1] .== 1) > Nfull₀
        @test all(μ[:, 1] .== μ[:, 3])
        @test all(isfinite, cpu_interior(model.velocities.u))
    end
end
