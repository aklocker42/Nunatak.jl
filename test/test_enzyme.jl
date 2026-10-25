using Enzyme
using Nunatak: ice_sheet_velocities, ice_sheet_auxiliary_fields, update_diva_coefficients!, compute_velocity_profile!

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

function grounded_fraction_cost(φ, φᵘ, φᵛ, constants, H, b)
    fill_halo_regions!(H)
    Nunatak.compute_grounded_fractions!(φ, φᵘ, φᵛ, SubgridGroundedFraction(), H, b, constants)
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

function sum_of_column_squares(f)
    Nx, Ny, Nz = size(f)
    J = zero(eltype(f))
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds J += f[i, j, k]^2
    end
    return J
end

function diva_velocity_cost(velocities, picard, solver, τ, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, φᵘ, φᵛ, N, βᵇᵘ, βᵇᵛ)
    (; u, v, uᵇ, vᵇ, uσ, vσ) = velocities
    for field in (u, v, ησ, F₂, βᵇᵘ, βᵇᵛ)
        fill!(field, 0)
    end
    fill_halo_regions!((rheology.rate_factor, friction.coefficient))
    coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, φᵘ, φᵛ, N, βᵇᵘ, βᵇᵛ, uᵇ, vᵇ)
    picard_solve!(VelocityVector(u, v), picard, solver, update_diva_coefficients!, τ, coefficients...)
    fill_halo_regions!(ησ)
    compute_velocity_profile!(velocities, βᵇᵘ, βᵇᵛ, F₂, ησ, H)
    return sum_of_squares(u) + sum_of_squares(v) + sum_of_column_squares(uσ) + sum_of_column_squares(vσ)
end

# Sliding and deformation of comparable size, so the gradient runs through both βᵇ and the shear integrals.
function diva_problem(; Lx = 8e4, Ly = 6e4, τ₀ = 1e4)
    grid = RectilinearGrid(size = (8, 6), x = (0, Lx), y = (0, Ly), topology = (Bounded, Periodic, Flat))
    σgrid = SigmaGrid(grid; Nσ = 6)
    velocities = ice_sheet_velocities(grid, σgrid)
    (; ησ, η̄H, η̄Hᶠᶠ, F₂, βᵘ, βᵛ, βᵇᵘ, βᵇᵛ) = ice_sheet_auxiliary_fields(grid, σgrid)

    H = CenterField(grid)
    set!(H, (x, y) -> 1000 + x / 1e2 + 100 * sin(2π * y / Ly))
    fill_halo_regions!(H)
    φᵘ, φᵛ = XFaceField(grid), YFaceField(grid)
    set!(φᵘ, 1)
    set!(φᵛ, 1)

    Az = (Lx / 8) * (Ly / 6)
    τ = VelocityVector(grid)
    set!(τ.u, (x, y) -> τ₀ * Az * sin(π * x / Lx) * (1 + sin(2π * y / Ly) / 5))
    set!(τ.v, (x, y) -> τ₀ * Az * cos(π * x / Lx) / 5)

    rheology = GlenFlowLaw(grid; rate_factor = (x, y) -> 1e-21 * (1 + cos(2π * y / Ly) / 5), strain_rate_regularization = 1e-16)
    friction = WeertmanFriction(grid; coefficient = (x, y) -> 1e6 * (1 + x / Lx))

    solver = VelocityConjugateGradientSolver(ssa_operator!, grid; iteration_control = Convergence(; reltol = 1e-14, maxiter = 10_000))
    picard = PicardIteration(grid; iteration_control = FixedIterations(3))

    return (velocities, picard, solver, τ, η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, φᵘ, φᵛ, nothing, βᵇᵘ, βᵇᵛ)
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
        constants = IceSheetConstants(grounding_grid)
        ρ = constants.ice_density / constants.ocean_density
        H, b = CenterField(grounding_grid), CenterField(grounding_grid)
        set!(H, (x, y) -> 1000 + 10x)
        set!(b, (x, y) -> ρ * (30 * (6 + sin(2π * y / 8) - x) - 1000 - 10x))
        fill_halo_regions!(b)
        φ, φᵘ, φᵛ = CenterField(grounding_grid), XFaceField(grounding_grid), YFaceField(grounding_grid)

        dH = Enzyme.make_zero(H)
        autodiff(set_runtime_activity(Reverse), grounded_fraction_cost, Active,
                 Duplicated(φ, Enzyme.make_zero(φ)), Duplicated(φᵘ, Enzyme.make_zero(φᵘ)), Duplicated(φᵛ, Enzyme.make_zero(φᵛ)),
                 Const(constants), Duplicated(H, dH), Const(b))

        for index in ((7, 2), (6, 5), (7, 7))
            fd = centred_difference(() -> grounded_fraction_cost(φ, φᵘ, φᵛ, constants, H, b), H, index, 1e-3)
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

    @testset "DIVA Picard iterations through update_diva_coefficients! and solve!" begin
        problem = diva_problem()
        velocities, picard, solver, τ = problem[1:4]
        rheology, friction = problem[12:13]
        shadows = map(Enzyme.make_zero, problem)
        arguments = map(problem, shadows) do value, shadow
            value isa Union{Nothing, PicardIteration, VelocityConjugateGradientSolver} ? Const(value) : Duplicated(value, shadow)
        end
        autodiff(set_runtime_activity(Reverse), diva_velocity_cost, Active, arguments...)

        cost() = diva_velocity_cost(problem...)
        dA, dC = shadows[12].rate_factor, shadows[13].coefficient
        for (name, field, shadow, index) in (("A", rheology.rate_factor, dA, (3, 2)),
                                             ("A", rheology.rate_factor, dA, (6, 5)),
                                             ("C", friction.coefficient, dC, (4, 3)),
                                             ("C", friction.coefficient, dC, (7, 6)))
            fd = centred_difference(cost, field, index, 1e-4 * field[index..., 1])
            ad = shadow[index..., 1]
            @info "∂J/∂$name through DIVA" index ad fd
            @test abs(ad - fd) < 1e-6 * abs(fd)
        end
    end
end
