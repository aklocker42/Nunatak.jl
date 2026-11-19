using Enzyme
using Nunatak: ice_sheet_geometry, ice_sheet_auxiliary_fields

function triangle_velocity_cost(x, solver, geometry, auxiliary, rheology, friction, constants, u_obs)
    set!(x.u, 1e-6)
    set!(x.v, 0)
    solve_stress_balance!(x, solver, geometry, auxiliary, rheology, friction, constants)
    J = zero(eltype(u_obs))
    for n in eachindex(u_obs)
        @inbounds J += (x.u.data[n] - u_obs[n])^2 + x.v.data[n]^2
    end
    return J / 2
end

@testset "Enzyme gradient of a triangle-mesh velocity misfit" begin
    N, L = 4, 40e3
    mesh = split_square_mesh(N, L; constrained = ((x, y) -> x == 0 || x == L, (x, y) -> y == 0 || y == L))
    geometry, auxiliary = ice_sheet_geometry(mesh), ice_sheet_auxiliary_fields(mesh)
    set!(geometry.H, (x, y) -> 1000 + x / 1e2 + 50 * sin(2π * y / L))
    rheology = GlenFlowLaw(mesh; rate_factor = 1e-24)
    friction = WeertmanFriction(mesh; coefficient = 1e4)
    set!(friction.coefficient, (x, y) -> 1e4 * (1 + x / 2L))
    constants = IceSheetConstants(mesh)
    solver = StressBalanceSolver(mesh; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 1000),
                                 picard_iteration_control = FixedIterations(4))
    u_obs = [1e-6 * x / L for x in mesh.x]
    x = VelocityVector(mesh)

    dfriction = Enzyme.make_zero(friction)
    autodiff(set_runtime_activity(Reverse), triangle_velocity_cost, Active,
             Duplicated(x, Enzyme.make_zero(x)), Const(solver), Duplicated(geometry, Enzyme.make_zero(geometry)),
             Duplicated(auxiliary, Enzyme.make_zero(auxiliary)), Const(rheology), Duplicated(friction, dfriction),
             Const(constants), Const(u_obs))
    gradient = copy(interior(dfriction.coefficient))

    C = interior(friction.coefficient)
    for n in findall(.!mesh.constrained_u)
        C₀ = C[n]
        δ = 1e-4 * C₀
        C[n] = C₀ + δ
        J⁺ = triangle_velocity_cost(x, solver, geometry, auxiliary, rheology, friction, constants, u_obs)
        C[n] = C₀ - δ
        J⁻ = triangle_velocity_cost(x, solver, geometry, auxiliary, rheology, friction, constants, u_obs)
        C[n] = C₀
        @test gradient[n] ≈ (J⁺ - J⁻) / 2δ rtol = 1e-5
    end
    @test any(!iszero, gradient)
end
