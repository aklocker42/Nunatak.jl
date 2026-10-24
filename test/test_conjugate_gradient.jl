function allocations_after_warmup(f, args...)
    f(args...)
    return @allocated f(args...)
end


@testset "Velocity conjugate gradient" begin
    grid = toy_grid(arch)
    ηH = toy_viscosity(grid)
    b = toy_right_hand_side(grid)

    solver = VelocityConjugateGradientSolver(toy_operator!, grid;
                                             iteration_control = Convergence(; reltol = 1e-12, maxiter = 500))
    x = VelocityVector(grid)
    solve!(x, solver, b, ηH)

    Ax = VelocityVector(grid)
    fill_halo_regions!(x)
    toy_operator!(Ax, x, ηH)
    residual = sqrt(sum(abs2, interior(Ax.u) .- interior(b.u)) + sum(abs2, interior(Ax.v) .- interior(b.v)))
    @test 0 < solver.iteration < 500
    @test residual < 1e-10 * norm(b)

    λ = VelocityVector(grid)
    adjoint_solve!(λ, solver, b, ηH)
    @test Array(interior(λ.u)) ≈ Array(interior(x.u))

    y = toy_right_hand_side(grid)
    set!(y.u, (x, y) -> cos(6π * y))
    Ay = VelocityVector(grid)
    fill_halo_regions!(y)
    toy_operator!(Ay, y, ηH)
    @test dot(Ax, y) ≈ dot(x, Ay)

    jacobi = JacobiPreconditioner(VelocityVector(grid))
    interior(jacobi.diagonal.u) .= 1 .+ interior(ηH) .+ circshift(interior(ηH), (1, 0, 0))
    interior(jacobi.diagonal.v) .= 1 .+ interior(ηH) .+ circshift(interior(ηH), (0, 1, 0))
    preconditioned = VelocityConjugateGradientSolver(toy_operator!, grid; preconditioner = jacobi,
                                                     iteration_control = Convergence(; reltol = 1e-12, maxiter = 500))
    xᴾ = VelocityVector(grid)
    solve!(xᴾ, preconditioned, b, ηH)
    @test Array(interior(xᴾ.u)) ≈ Array(interior(x.u))
    @test Array(interior(xᴾ.v)) ≈ Array(interior(x.v))

    fixed = VelocityConjugateGradientSolver(toy_operator!, grid; iteration_control = FixedIterations(4))
    picard = PicardIteration(grid; iteration_control = FixedIterations(3))
    x³ = VelocityVector(grid)
    picard_solve!(x³, picard, fixed, toy_update_coefficients!, b, ηH)
    @test fixed.iteration == 4

    allocations = allocations_after_warmup(picard_solve!, x³, picard, fixed, toy_update_coefficients!, b, ηH)
    @info "Picard(FixedIterations(3)) × CG(FixedIterations(4)) allocations" allocations
    arch isa CPU && @test allocations == 0
end
