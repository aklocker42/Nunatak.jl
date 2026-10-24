using Enzyme

function sum_of_squares(f)
    Nx, Ny, _ = size(f)
    J = zero(eltype(f))
    for j in 1:Ny, i in 1:Nx
        @inbounds J += f[i, j, 1]^2
    end
    return J
end

function driving_stress_cost(τ, H, s)
    fill_halo_regions!(H)
    toy_driving_stress!(τ, H, s)
    return sum_of_squares(τ)
end

function grounded_fraction_cost(φ, φᵘ, φᵛ, grounding, H, b)
    fill_halo_regions!(H)
    Nunatak.compute_grounded_fractions!(φ, φᵘ, φᵛ, grounding, H, b)
    return sum_of_squares(φ) + sum_of_squares(φᵘ) + sum_of_squares(φᵛ)
end

function velocity_cost(x, solver, b, ηH)
    fill_halo_regions!(ηH)
    fill!(x, 0)
    solve!(x, solver, b, ηH)
    return sum_of_squares(x.u) + sum_of_squares(x.v)
end

function ssa_velocity_cost(x, solver, b, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ)
    fill_halo_regions!((η̄H, η̄Hᶠᶠ, βᵘ, βᵛ))
    fill!(x, 0)
    solve!(x, solver, b, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ)
    return sum_of_squares(x.u) + sum_of_squares(x.v)
end

function centred_difference(cost, field, index, h)
    i, j = index
    field[i, j, 1] += h
    J⁺ = cost()
    field[i, j, 1] -= 2h
    J⁻ = cost()
    field[i, j, 1] += h
    return (J⁺ - J⁻) / 2h
end

@testset "Enzyme gradients" begin
    grid = toy_grid()

    @testset "Driving stress kernel through launch!" begin
        H = Field{Center, Center, Nothing}(grid)
        s = Field{Center, Center, Nothing}(grid)
        τ = Field{Face, Center, Nothing}(grid)
        set!(H, (x, y) -> 1000 + 200 * sin(2π * x) * cos(2π * y))
        set!(s, (x, y) -> 1e-3 * cos(2π * x) + 1e-4 * sin(2π * y))
        fill_halo_regions!(s)

        dH = Enzyme.make_zero(H)
        dτ = Enzyme.make_zero(τ)
        autodiff(set_runtime_activity(Reverse), driving_stress_cost, Active,
                 Duplicated(τ, dτ), Duplicated(H, dH), Const(s))

        for index in ((1, 1), (3, 5), (8, 2))
            fd = centred_difference(() -> driving_stress_cost(τ, H, s), H, index, 1.0)
            ad = dH[index..., 1]
            @info "∂J/∂H" index ad fd
            @test abs(ad - fd) < 1e-6 * abs(fd)
        end
    end

    @testset "Grounded fraction kernels through launch!" begin
        grounding_grid = RectilinearGrid(size = (12, 8), x = (0, 12), y = (0, 8), topology = (Bounded, Periodic, Flat))
        grounding = SubgridGroundedFraction(grounding_grid)
        ρ = grounding.ice_density / grounding.ocean_density
        H, b = CenterField(grounding_grid), CenterField(grounding_grid)
        set!(H, (x, y) -> 1000 + 10x)
        set!(b, (x, y) -> ρ * (30 * (6 + sin(2π * y / 8) - x) - 1000 - 10x))
        fill_halo_regions!(b)
        φ, φᵘ, φᵛ = CenterField(grounding_grid), XFaceField(grounding_grid), YFaceField(grounding_grid)

        dH = Enzyme.make_zero(H)
        autodiff(set_runtime_activity(Reverse), grounded_fraction_cost, Active,
                 Duplicated(φ, Enzyme.make_zero(φ)), Duplicated(φᵘ, Enzyme.make_zero(φᵘ)), Duplicated(φᵛ, Enzyme.make_zero(φᵛ)),
                 Const(grounding), Duplicated(H, dH), Const(b))

        for index in ((7, 2), (6, 5), (7, 7))
            fd = centred_difference(() -> grounded_fraction_cost(φ, φᵘ, φᵛ, grounding, H, b), H, index, 1e-3)
            ad = dH[index..., 1]
            @info "∂J/∂H through φ" index ad fd
            @test abs(ad - fd) < 1e-6 * abs(fd)
        end
    end

    @testset "Custom adjoint of solve!" begin
        ηH = toy_viscosity(grid)
        b = toy_right_hand_side(grid)
        x = VelocityVector(grid)
        solver = VelocityConjugateGradientSolver(toy_operator!, grid;
                                                 iteration_control = Convergence(; reltol = 1e-14, maxiter = 1000))

        dηH = Enzyme.make_zero(ηH)
        db = Enzyme.make_zero(b)
        dx = Enzyme.make_zero(x)
        autodiff(set_runtime_activity(Reverse), velocity_cost, Active,
                 Duplicated(x, dx), Const(solver), Duplicated(b, db), Duplicated(ηH, dηH))

        cost() = velocity_cost(x, solver, b, ηH)

        for index in ((1, 1), (4, 6), (7, 3))
            fd = centred_difference(cost, ηH, index, 1e-4)
            ad = dηH[index..., 1]
            @info "∂J/∂ηH" index ad fd
            @test abs(ad - fd) < 1e-6 * abs(fd)
        end

        fd = centred_difference(cost, b.u, (2, 3), 1e-4)
        ad = db.u[2, 3, 1]
        @info "∂J/∂bᵘ" ad fd
        @test abs(ad - fd) < 1e-6 * abs(fd)
    end

    @testset "Custom adjoint of solve! through the SSA operator" begin
        grid = RectilinearGrid(size = (8, 6), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat))
        coefficients = ssa_coefficients(grid)
        b = toy_right_hand_side(grid)
        x = VelocityVector(grid)
        solver = VelocityConjugateGradientSolver(ssa_operator!, grid;
                                                 iteration_control = Convergence(; reltol = 1e-14, maxiter = 10_000))

        shadows = map(Enzyme.make_zero, coefficients)
        autodiff(set_runtime_activity(Reverse), ssa_velocity_cost, Active,
                 Duplicated(x, Enzyme.make_zero(x)), Const(solver), Duplicated(b, Enzyme.make_zero(b)),
                 map(Duplicated, coefficients, shadows)...)

        # Free-slip walls leave the y-mean v mode resisted only by drag, so Float64 differences drown in roundoff.
        big_grid = RectilinearGrid(BigFloat; size = (8, 6), x = (0, 1), y = (0, 1), topology = (Bounded, Periodic, Flat))
        big_coefficients = ssa_coefficients(big_grid)
        big_b = toy_right_hand_side(big_grid)
        big_x = VelocityVector(big_grid)
        big_solver = VelocityConjugateGradientSolver(ssa_operator!, big_grid;
                                                     iteration_control = Convergence(; reltol = big"1e-40", maxiter = 10_000))
        cost() = ssa_velocity_cost(big_x, big_solver, big_b, big_coefficients...)

        dη̄H, dη̄Hᶠᶠ, dβᵘ, dβᵛ = shadows
        for (name, field, shadow, index) in (("η̄H", big_coefficients[1], dη̄H, (3, 2)),
                                             ("η̄Hᶠᶠ", big_coefficients[2], dη̄Hᶠᶠ, (4, 5)),
                                             ("βᵘ", big_coefficients[3], dβᵘ, (5, 3)),
                                             ("βᵛ", big_coefficients[4], dβᵛ, (2, 4)))
            fd = centred_difference(cost, field, index, big"1e-15")
            ad = shadow[index..., 1]
            @info "∂J/∂$name" index ad fd
            @test abs(ad - fd) < 1e-6 * abs(fd)
        end
    end
end
