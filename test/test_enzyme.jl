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

        η̄H, η̄Hᶠᶠ, βᵘ, βᵛ = coefficients
        dη̄H, dη̄Hᶠᶠ, dβᵘ, dβᵛ = shadows
        cost() = ssa_velocity_cost(x, solver, b, coefficients...)

        for (name, field, shadow, index) in (("η̄H", η̄H, dη̄H, (3, 2)), ("η̄Hᶠᶠ", η̄Hᶠᶠ, dη̄Hᶠᶠ, (4, 5)),
                                             ("βᵘ", βᵘ, dβᵘ, (5, 3)), ("βᵛ", βᵛ, dβᵛ, (2, 4)))
            fd = centred_difference(cost, field, index, 1e-4)
            ad = shadow[index..., 1]
            @info "∂J/∂$name" index ad fd
            @test abs(ad - fd) < 1e-6 * abs(fd)
        end
    end
end
