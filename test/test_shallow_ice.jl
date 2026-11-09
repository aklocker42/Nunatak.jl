using Nunatak: compute_strain_heating!, layer_thickness
using Oceananigans.Grids: halo_size
using Oceananigans.Advection: Centered
using Oceananigans.Diagnostics: cell_diffusion_timescale

# Uniform slab of thickness H₀ on a bed sloping down in x with slope α: the shallow-ice solution is uniform in x.
slab_friction(::Nothing, grid) = nothing
slab_friction(build, grid) = build(grid)

# along = :y turns the slab so that it slopes down in y on a (Periodic, Bounded) grid.
function shallow_ice_slab(; Nσ = 4, H₀ = 1000, α = 1e-3, A = 1e-24, friction = nothing, Δx = 1e3, along = :x)
    grid = along == :x ?
        RectilinearGrid(arch; size = (16, 4), halo = (4, 4), x = (0, 16Δx), y = (0, 4Δx), topology = (Bounded, Periodic, Flat)) :
        RectilinearGrid(arch; size = (4, 16), halo = (4, 4), x = (0, 4Δx), y = (0, 16Δx), topology = (Periodic, Bounded, Flat))
    bed = along == :x ? (x, y) -> -α * x : (x, y) -> -α * y
    model = IceSheetModel(grid; Nσ, stress_balance = ShallowIce(), rheology = GlenFlowLaw(grid; rate_factor = A),
                          friction = slab_friction(friction, grid), bed, constants = IceSheetConstants(grid; ice_density = 910))
    set!(model; H = H₀)
    return model
end

slab_driving_stress(model, H₀, α) = model.constants.ice_density * model.constants.gravitational_acceleration * H₀ * α

# Halfar (1983) similarity solution for n = 3: H₀ (t₀/t)^(1/9) [1 - ((t₀/t)^(1/18) r / R₀)^(4/3)]^(3/7).
function halfar_thickness(r, t, t₀, H₀, R₀)
    ξ = (t₀ / t)^(1 / 18) * r / R₀
    return H₀ * (t₀ / t)^(1 / 9) * max(0, 1 - ξ^(4 / 3))^(3 / 7)
end

@testset "Shallow-ice approximation" begin
    @testset "Default solver, advection and friction check" begin
        model = shallow_ice_slab()
        @test model.velocity_solver isa ShallowIceSolver
        @test model.advection isa Centered
        grid = model.grid
        @test_throws ArgumentError IceSheetModel(grid; stress_balance = ShallowIce(), rheology = GlenFlowLaw(grid; rate_factor = 1e-24),
                                                 friction = RegularizedCoulombFriction(grid; coefficient = 1e4, coulomb_coefficient = 0.5))
        @test IceSheetModel(grid; stress_balance = ShallowIce()).velocity_solver === nothing
        rheology = GlenFlowLaw(grid; rate_factor = 1e-24)
        @test_throws ArgumentError IceSheetModel(grid; stress_balance = ShallowIce(), rheology, velocity_solver = StressBalanceSolver(grid))
        @test_throws ArgumentError IceSheetModel(grid; rheology, velocity_solver = ShallowIceSolver())
    end

    @testset "Slab on a uniform slope matches the analytic profile and mean" begin
        H₀, α, A = 1000, 1e-3, 1e-24
        model = shallow_ice_slab(; H₀, α, A)
        τ = slab_driving_stress(model, H₀, α)
        U = 2A * τ^3 * H₀
        ζ = Array(znodes(model.σgrid, Center()))
        uσ = Array(interior(model.velocities.uσ, 8, 2, :))
        @test uσ ≈ U .* (1 .- (1 .- ζ) .^ 4) ./ 4 rtol = 1e-12
        u = Array(interior(model.velocities.u, :, 2, 1))
        @test all(isapprox.(u[2:end-1], U / 5; rtol = 1e-12))
        @test u[1] == 0 && u[end] == 0
        @test maximum(abs, Array(interior(model.velocities.v))) == 0

        turned = shallow_ice_slab(; H₀, α, A, along = :y)
        v = Array(interior(turned.velocities.v, 2, :, 1))
        @test all(isapprox.(v[2:end-1], U / 5; rtol = 1e-12))
        @test v[1] == 0 && v[end] == 0
        @test maximum(abs, Array(interior(turned.velocities.u))) == 0
    end

    @testset "Weertman sliding balances the driving stress" begin
        H₀, α, C = 1000, 1e-3, 1e10
        model = shallow_ice_slab(; H₀, α, friction = grid -> WeertmanFriction(grid; coefficient = C, exponent = 1))
        τ = slab_driving_stress(model, H₀, α)
        uᵇ = Array(interior(model.velocities.uᵇ, 8, 2, 1))[1]
        @test uᵇ ≈ τ / C rtol = 1e-12
        @test Array(interior(model.auxiliary.βᵘ, 8, 2, 1))[1] * uᵇ ≈ τ rtol = 1e-6
        uσ = Array(interior(model.velocities.uσ, 8, 2, :))
        @test all(uσ .> uᵇ)

        frozen = shallow_ice_slab(; H₀, α, friction = grid -> WeertmanFriction(grid; coefficient = 0, exponent = 1))
        @test maximum(abs, Array(interior(frozen.velocities.uᵇ))) == 0
        u = Array(interior(frozen.velocities.u, :, 2, 1))
        @test all(isfinite, u) && all(isapprox.(u[2:end-1], 2 * 1e-24 * τ^3 * H₀ / 5; rtol = 1e-12))
    end

    @testset "Column strain heating equals the work of the driving stress on the shear" begin
        H₀, α, A = 1000, 1e-3, 1e-24
        model = shallow_ice_slab(; Nσ = 8, H₀, α, A)
        σgrid = model.σgrid
        Φ = CenterField(σgrid)
        compute_strain_heating!(Φ, model, model.stress_balance)
        Φᵢ = Array(interior(Φ, 8, 2, :))
        Δζ = [layer_thickness(k, on_architecture(CPU(), σgrid)) for k in 1:size(σgrid, 3)]
        ū = Array(interior(model.velocities.u, 8, 2, 1))[1]
        @test H₀ * sum(Φᵢ .* Δζ) ≈ slab_driving_stress(model, H₀, α) * ū rtol = 1e-12
        @test Array(interior(Φ, 1, 2, :)) ≈ Φᵢ rtol = 1e-12
        @test Array(interior(Φ, 16, 2, :)) ≈ Φᵢ rtol = 1e-12
    end

    @testset "Diffusion time scale" begin
        H₀, α = 1000, 1e-3
        model = shallow_ice_slab(; H₀, α)
        ū = Array(interior(model.velocities.u, 8, 2, 1))[1]
        @test cell_diffusion_timescale(model) ≈ 1e3^2 / (ū * H₀ / α) rtol = 1e-12
        grid = model.grid
        ssa = IceSheetModel(grid; rheology = GlenFlowLaw(grid; rate_factor = 1e-24))
        @test cell_diffusion_timescale(ssa) == Inf
    end

    @testset "Enthalpy thermodynamics with an Arrhenius rate factor" begin
        H₀, α = 1000, 1e-3
        grid = RectilinearGrid(arch; size = (16, 4), halo = (4, 4), x = (0, 1.6e4), y = (0, 4e3), topology = (Bounded, Periodic, Flat))
        model = IceSheetModel(grid; stress_balance = ShallowIce(), rheology = GlenFlowLaw(grid; rate_factor = ArrheniusRateFactor(grid)),
                              friction = WeertmanFriction(grid; coefficient = 1e10, exponent = 1), bed = (x, y) -> -α * x,
                              thermodynamics = EnthalpyThermodynamics(grid; surface_temperature = 263.15, geothermal_flux = 0.05),
                              constants = IceSheetConstants(grid; ice_density = 910))
        set!(model; H = H₀)
        for _ in 1:10
            time_step!(model, min(cell_diffusion_timescale(model) / 4, 3.15e7))
        end
        Φ = Array(interior(model.thermodynamics.state.Φ))
        E = Array(interior(model.thermodynamics.state.E))
        u = Array(interior(model.velocities.u, :, 2, 1))
        @test all(isfinite, Φ) && all(isfinite, E)
        @test minimum(Φ) ≥ 0
        @test all(Φ[2:end-1, :, :] .> 0)
        @test all(isfinite, u) && all(u[2:end-1] .> 0)

        periodic = RectilinearGrid(arch; size = (4, 16), halo = (4, 4), x = (0, 4e3), y = (0, 1.6e4), topology = (Periodic, Bounded, Flat))
        model = IceSheetModel(periodic; stress_balance = ShallowIce(), rheology = GlenFlowLaw(periodic; rate_factor = ArrheniusRateFactor(periodic)),
                              bed = (x, y) -> -α * y, thermodynamics = EnthalpyThermodynamics(periodic; surface_temperature = 263.15, geothermal_flux = 0.05),
                              constants = IceSheetConstants(periodic; ice_density = 910))
        set!(model; H = H₀)
        time_step!(model, 3.15e7)
        A = Array(parent(model.rheology.rate_factor.field))
        Hx = halo_size(model.σgrid)[1]
        @test A[Hx, :, :] == A[Hx + 4, :, :]
    end

    @testset "Halfar dome: conserves volume and follows the similarity solution" begin
        spy = 31556926.0
        H₀, R₀, A, ρᵢ, g = 3600.0, 750e3, 1e-16 / spy, 910, 9.81
        Γ = 2A * (ρᵢ * g)^3 / 5
        t₀ = (7 / 4)^3 * R₀^4 / (18Γ * H₀^7)
        Nx, L = 49, 1225e3
        grid = RectilinearGrid(arch; size = (Nx, Nx), halo = (4, 4), x = (-L, L), y = (-L, L), topology = (Bounded, Bounded, Flat))
        model = IceSheetModel(grid; stress_balance = ShallowIce(), rheology = GlenFlowLaw(grid; rate_factor = A),
                              constants = IceSheetConstants(grid; ice_density = ρᵢ, gravitational_acceleration = g))
        model.clock.time = t₀
        set!(model; H = (x, y) -> halfar_thickness(hypot(x, y), t₀, t₀, H₀, R₀))
        V₀ = sum(Array(interior(model.geometry.H)))
        while model.clock.time < 2t₀
            Δt = min(cell_diffusion_timescale(model) / 4, 2t₀ - model.clock.time)
            time_step!(model, Δt)
        end
        H = Array(interior(model.geometry.H, :, :, 1))
        x = Array(xnodes(grid, Center()))
        exact = [halfar_thickness(hypot(xᵢ, yⱼ), 2t₀, t₀, H₀, R₀) for xᵢ in x, yⱼ in x]
        interior_ice = [hypot(xᵢ, yⱼ) ≤ 600e3 for xᵢ in x, yⱼ in x]
        @test sum(H) ≈ V₀ rtol = 1e-12
        @test minimum(H) ≥ 0
        @test abs(H[25, 25] - exact[25, 25]) / exact[25, 25] < 1e-3
        @test maximum(abs, (H .- exact)[interior_ice] ./ exact[interior_ice]) < 0.01
        @test maximum(abs, H .- exact) / H₀ < 0.08
        @test H ≈ H' rtol = 1e-12
        @test H ≈ reverse(H, dims = 1) rtol = 1e-12

        Φ = CenterField(model.σgrid)
        compute_strain_heating!(Φ, model, model.stress_balance)
        Φᵢ = Array(interior(Φ))
        @test Φᵢ ≈ reverse(Φᵢ, dims = 1) rtol = 1e-10
    end
end
