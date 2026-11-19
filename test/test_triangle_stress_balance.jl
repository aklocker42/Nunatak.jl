using Nunatak: triangle_ssa_operator!, compute_element_viscosity!, compute_driving_stress!, ice_sheet_geometry, ice_sheet_auxiliary_fields

function random_node_vector(mesh, seed)
    v = VelocityVector(mesh)
    set!(v.u, (x, y) -> sin(seed * x + 2y) + cos(3x - seed * y))
    set!(v.v, (x, y) -> cos(seed * x * y) - sin(x + seed))
    return v
end

@testset "Stress-balance operator on a triangle mesh" begin
    N, L = 8, 2.0
    mesh = split_square_mesh(N, L; arch, constrained = ((x, y) -> x == 0, (x, y) -> y == 0))
    η̄H, β = ElementField(mesh), NodeField(mesh)
    set!(η̄H, (x, y) -> 1 + x * y)
    set!(β, (x, y) -> 0.1 + x / 4)

    p, q = random_node_vector(mesh, 1.3), random_node_vector(mesh, 2.7)
    Ap, Aq = VelocityVector(mesh), VelocityVector(mesh)
    triangle_ssa_operator!(Ap, p, η̄H, β)
    triangle_ssa_operator!(Aq, q, η̄H, β)
    @test dot(Ap, q) ≈ dot(p, Aq) rtol = 1e-12
    @test dot(Ap, p) > 0

    xc, yc = Array(mesh.x), Array(mesh.y)
    constrained_u, constrained_v = Array(mesh.constrained_u), Array(mesh.constrained_v)
    @test Array(interior(Ap.u))[constrained_u] == Array(interior(p.u))[constrained_u]
    @test Array(interior(Ap.v))[constrained_v] == Array(interior(p.v))[constrained_v]

    # Without viscosity the operator is the lumped drag
    set!(η̄H, 0)
    set!(β, 1)
    triangle_ssa_operator!(Ap, p, η̄H, β)
    free = .!constrained_u
    @test Array(interior(Ap.u))[free] ≈ Array(mesh.node_area)[free] .* Array(interior(p.u))[free]

    # A uniform stress has no divergence away from the boundary
    unconstrained = split_square_mesh(N, L; arch, constrained = (falses(length(xc)), falses(length(xc))))
    η̄H, β = set!(ElementField(unconstrained), 3), NodeField(unconstrained)
    u = VelocityVector(unconstrained)
    set!(u.u, (x, y) -> x)
    set!(u.v, (x, y) -> x - 2y)
    triangle_ssa_operator!(Ap, u, η̄H, β)
    interior_nodes = (0 .< xc .< L) .& (0 .< yc .< L)
    @test maximum(abs, Array(interior(Ap.u))[interior_nodes]) < 1e-12
    @test maximum(abs, Array(interior(Ap.v))[interior_nodes]) < 1e-12

    # Ice-free nodes get identity rows
    set!(η̄H, 0)
    triangle_ssa_operator!(Ap, p, η̄H, β)
    @test Array(interior(Ap.u)) == Array(interior(p.u))

    # The diagonal preconditioner divides by the operator diagonal
    η̄H, β = set!(ElementField(mesh), 2), set!(NodeField(mesh), 0.5)
    z, e, Ae = VelocityVector(mesh), VelocityVector(mesh), VelocityVector(mesh)
    fill!(p, 1)
    Nunatak.precondition!(z, StressBalanceDiagonalPreconditioner(), p, η̄H, β)
    n = findfirst(interior_nodes .& free)
    interior(e.u)[n:n] .= 1
    triangle_ssa_operator!(Ae, e, η̄H, β)
    @test Array(interior(z.u))[n] ≈ 1 / Array(interior(Ae.u))[n]
end

@testset "Element viscosity and driving stress on a triangle mesh" begin
    mesh = split_square_mesh(4, 1.0; arch, constrained = (falses(25), falses(25)))
    rheology = GlenFlowLaw(mesh; rate_factor = 2, exponent = 3, strain_rate_regularization = 0)
    u, H, η̄H = VelocityVector(mesh), set!(NodeField(mesh), (x, y) -> 1 + x), ElementField(mesh)
    set!(u.u, (x, y) -> 3x)
    set!(u.v, (x, y) -> 4y)
    compute_element_viscosity!(η̄H, rheology, u.u, u.v, H)
    ε̇ₑ² = 9 + 16 + 12
    H̄ = [1 + x for x in Array(interior(set!(ElementField(mesh), (x, y) -> x)))]
    @test Array(interior(η̄H)) ≈ 2^(-1 / 3) * ε̇ₑ²^(-1 / 3) / 2 .* H̄

    # ∫ H ∇s φ over the whole square is exact for P1 H and s
    s = set!(NodeField(mesh), (x, y) -> 2x + y)
    τ = VelocityVector(mesh)
    compute_driving_stress!(τ, H, s, 1, 1, 1)
    @test sum(Array(interior(τ.u))) ≈ -2 * 1.5
    @test sum(Array(interior(τ.v))) ≈ -1.5
end

@testset "Unconfined shelf on a triangle mesh matches the analytic spreading rate" begin
    N, L, H₀, A = 8, 40e3, 500, 1e-24
    front = permutedims(hcat(N + 1 .+ (N + 1) .* (0:N-1), N + 1 .+ (N + 1) .* (1:N)))
    mesh = split_square_mesh(N, L; arch, front_segments = front,
                             constrained = ((x, y) -> x == 0, (x, y) -> y == 0 || y == L))
    geometry, auxiliary = ice_sheet_geometry(mesh), ice_sheet_auxiliary_fields(mesh)
    set!(geometry.H, H₀)
    set!(geometry.b, -2000)
    constants = IceSheetConstants(mesh)
    rheology = GlenFlowLaw(mesh; rate_factor = A)
    solver = StressBalanceSolver(mesh; linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000),
                                 picard_iteration_control = Convergence(; reltol = 1e-10, maxiter = 500))
    velocity = VelocityVector(mesh)
    set!(velocity.u, (x, y) -> 1e-9 * x)
    solve_stress_balance!(velocity, solver, geometry, auxiliary, rheology, nothing, constants)

    ρ′ = constants.ice_density * (1 - constants.ice_density / constants.ocean_density)
    ε̇ = A * (ρ′ * constants.gravitational_acceleration * H₀ / 4)^3
    x = Array(mesh.x)
    @test 0 < solver.iteration < 500
    @test maximum(abs, Array(interior(velocity.u)) .- ε̇ .* x) / (ε̇ * L) < 1e-6
    @test maximum(abs, Array(interior(velocity.v))) < 1e-12 * ε̇ * L
end

@testset "Grounded slab on a triangle mesh matches the C-grid" begin
    N, L = 16, 64e3
    rate_factor, coefficient = 1e-24, 1e10
    thickness(x, y) = 1000 + x / 1e2

    grid = RectilinearGrid(arch; size = (N, N), halo = (4, 4), x = (0, L), y = (0, L), topology = (Bounded, Bounded, Flat))
    control = (linear_iteration_control = Convergence(; reltol = 1e-12, maxiter = 10_000),
               picard_iteration_control = Convergence(; reltol = 1e-10, maxiter = 500))
    model = IceSheetModel(grid; velocity_solver = StressBalanceSolver(grid; control...),
                          rheology = GlenFlowLaw(grid; rate_factor), friction = WeertmanFriction(grid; coefficient, exponent = 1),
                          bed = 0)
    set!(model; H = thickness)
    structured = Array(interior(model.velocities.u, :, N ÷ 2, 1))

    # Free-slip walls: normal velocity constrained on each wall
    mesh = split_square_mesh(N, L; arch, constrained = ((x, y) -> x == 0 || x == L, (x, y) -> y == 0 || y == L))
    geometry, auxiliary = ice_sheet_geometry(mesh), ice_sheet_auxiliary_fields(mesh)
    set!(geometry.H, thickness)
    velocity = VelocityVector(mesh)
    solve_stress_balance!(velocity, StressBalanceSolver(mesh; control...), geometry, auxiliary,
                          GlenFlowLaw(mesh; rate_factor), WeertmanFriction(mesh; coefficient, exponent = 1), IceSheetConstants(mesh))
    row = (N ÷ 2) * (N + 1) .+ (1:N+1)
    triangle = Array(interior(velocity.u))[row]
    difference = maximum(abs, triangle .- structured) / maximum(abs, structured)
    @info "Grounded slab, triangle vs C-grid" difference
    @test difference < 1e-2
    # One-way diagonals break the y-symmetry; the spurious v converges at second order (2.5e-4 of max u at N = 16)
    @test maximum(abs, Array(interior(velocity.v))) < 1e-3 * maximum(abs, structured)
end
