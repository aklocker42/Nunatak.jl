using Nunatak: linear_combination!, divide!, component_dot!

function cpu_mesh_diagonal(mesh)
    offsets, node_elements, local_index, elements = map(a -> on_architecture(CPU(), a),
        (mesh.node_element_offsets, mesh.node_elements, mesh.node_local_index, mesh.elements))
    area, ∂φ∂x, ∂φ∂y, node_area = map(a -> on_architecture(CPU(), a), (mesh.area, mesh.∂φ∂x, mesh.∂φ∂y, mesh.node_area))
    diagonal = ones(eltype(node_area), length(node_area))
    for n in eachindex(diagonal), position in offsets[n]:offsets[n+1]-1
        e, i = node_elements[position], local_index[position]
        diagonal[n] += area[e] * (∂φ∂x[i, e]^2 + ∂φ∂y[i, e]^2)
    end
    return diagonal
end

@testset "Triangle mesh" begin
    N, L = 6, 3.0
    mesh = split_square_mesh(N, L; arch)
    FT = eltype(mesh)
    @test FT == Float64
    @test Nunatak.architecture(mesh) == arch
    @test length(mesh.x) == (N + 1)^2
    @test size(mesh.elements) == (3, 2N^2)
    @test typeof(mesh.x) == typeof(on_architecture(arch, zeros(FT, 1)))
    @test typeof(mesh.elements) == typeof(on_architecture(arch, zeros(Int, 1, 1)))

    x, y, elements = map(a -> on_architecture(CPU(), a), (mesh.x, mesh.y, mesh.elements))
    area, ∂φ∂x, ∂φ∂y, node_area = map(a -> on_architecture(CPU(), a), (mesh.area, mesh.∂φ∂x, mesh.∂φ∂y, mesh.node_area))
    @test sum(area) ≈ L^2
    @test sum(node_area) ≈ L^2
    @test all(e -> Nunatak.element_signed_area(x, y, view(elements, :, e)) > 0, axes(elements, 2))

    f = 2x .+ 3y
    for e in axes(elements, 2)
        nodes = view(elements, :, e)
        @test sum(∂φ∂x[i, e] * f[nodes[i]] for i in 1:3) ≈ 2 atol = 1e-12
        @test sum(∂φ∂y[i, e] * f[nodes[i]] for i in 1:3) ≈ 3 atol = 1e-12
        @test sum(∂φ∂x[:, e]) ≈ 0 atol = 1e-12
        @test sum(∂φ∂y[:, e]) ≈ 0 atol = 1e-12
    end

    offsets, node_elements, local_index = map(a -> on_architecture(CPU(), a),
        (mesh.node_element_offsets, mesh.node_elements, mesh.node_local_index))
    @test length(node_elements) == 3 * size(elements, 2)
    @test offsets[end] == length(node_elements) + 1
    @test all(elements[local_index[p], node_elements[p]] == n for n in eachindex(x) for p in offsets[n]:offsets[n+1]-1)

    constrained_u, constrained_v = Array(mesh.constrained_u), Array(mesh.constrained_v)
    @test sum(constrained_u) == 4N
    @test constrained_u == constrained_v
    @test all(constrained_u .== ((x .== 0) .| (x .== L) .| (y .== 0) .| (y .== L)))
    @test size(mesh.front_segments) == (2, 0)

    @test sprint(show, mesh) == "TriangleMesh{Float64} on $(typeof(arch).name.name) with 49 nodes, 72 elements and 0 front segments"

    # Inflow at x = 0 constrained, free front along x = L
    right = findall((x .== L) .& (y .< L))
    front = permutedims(hcat(right, right .+ (N + 1)))
    custom = split_square_mesh(N, L; arch, constrained = ((x, y) -> x == 0, (x, y) -> x == 0 || y == 0 || y == L),
                               front_segments = front)
    @test sum(Array(custom.constrained_u)) == N + 1
    @test sum(Array(custom.constrained_v)) == 3N + 1
    @test size(custom.front_segments) == (2, N)
    @test Array(custom.front_normals) ≈ repeat([1.0, 0.0], 1, N)
    @test Array(custom.front_lengths) ≈ fill(L / N, N)

    @test_throws ArgumentError TriangleMesh(arch; x = [0.0, 1.0, 2.0], y = [0.0, 0.0, 0.0], elements = reshape([1, 2, 3], 3, 1))
end

@testset "Mesh fields and vectors" begin
    N, L = 6, 3.0
    mesh = split_square_mesh(N, L; arch)
    Nnodes = length(mesh.x)

    f = NodeField(mesh)
    @test length(f) == Nnodes
    @test parent(f) === interior(f)
    set!(f, (x, y) -> x - y)
    xc, yc = Array(mesh.x), Array(mesh.y)
    @test Array(interior(f)) ≈ xc .- yc
    set!(f, 2)
    @test all(Array(interior(f)) .== 2)
    g = ElementField(mesh)
    set!(g, (x, y) -> x + y)
    @test length(g) == 2N^2
    @test sum(Array(interior(g)) .* Array(mesh.area)) ≈ L^3   # ∫ (x + y) dA, exact for the centroid rule
    @test Nunatak.parameter_field(mesh, 3) isa NodeField
    @test Nunatak.parameter_field(mesh, f) === f
    @test Nunatak.parameter_field(mesh, nothing) === nothing
    @test fill_halo_regions!(f) === nothing

    u = VelocityVector(mesh)
    @test u.u isa NodeField && u.v isa NodeField
    fill!(u, 1)
    v = similar(u)
    set!(v.u, (x, y) -> x)
    set!(v.v, (x, y) -> y)
    w = VelocityVector(mesh)
    linear_combination!(w, 2, u, -1, v)
    @test Array(interior(w.u)) ≈ 2 .- xc
    @test Array(interior(w.v)) ≈ 2 .- yc
    @test dot(u, v) ≈ sum(xc) + sum(yc)
    @test norm(u) ≈ sqrt(2Nnodes)
    copyto!(w, v)
    divide!(w, w, u)
    @test Array(interior(w.u)) ≈ xc
    @test fill_halo_regions!(u) === nothing
    @test Nunatak.adapt(CPU(), u).u.mesh === nothing
end

@testset "Conjugate gradient on a triangle mesh" begin
    mesh = split_square_mesh(12, 1.0; arch)
    b = VelocityVector(mesh)
    set!(b.u, (x, y) -> sin(2π * x) * cos(4π * y))
    set!(b.v, (x, y) -> cos(2π * x) + x * y)

    mass = VelocityConjugateGradientSolver(toy_mesh_mass_operator!, mesh; iteration_control = Convergence(; reltol = 1e-12, maxiter = 50))
    x = VelocityVector(mesh)
    solve!(x, mass, b, mesh)
    @test Array(interior(x.u)) ≈ Array(interior(b.u)) ./ Array(mesh.node_area)
    @test mass.iteration <= 6

    solver = VelocityConjugateGradientSolver(toy_mesh_operator!, mesh; iteration_control = Convergence(; reltol = 1e-12, maxiter = 500))
    x = VelocityVector(mesh)
    solve!(x, solver, b, mesh)
    Ax = VelocityVector(mesh)
    toy_mesh_operator!(Ax, x, mesh)
    residual = sqrt(sum(abs2, Array(interior(Ax.u)) .- Array(interior(b.u))) + sum(abs2, Array(interior(Ax.v)) .- Array(interior(b.v))))
    @test 0 < solver.iteration < 500
    @test residual < 1e-10 * norm(b)

    λ = VelocityVector(mesh)
    adjoint_solve!(λ, solver, b, mesh)
    @test Array(interior(λ.u)) ≈ Array(interior(x.u))

    y = VelocityVector(mesh)
    set!(y.u, (x, y) -> cos(6π * y))
    set!(y.v, (x, y) -> x^2)
    Ay = VelocityVector(mesh)
    toy_mesh_operator!(Ay, y, mesh)
    @test dot(Ax, y) ≈ dot(x, Ay)
    @test dot(Ay, y) > 0

    jacobi = JacobiPreconditioner(VelocityVector(mesh))
    diagonal = cpu_mesh_diagonal(mesh)
    set!(jacobi.diagonal.u, diagonal)
    set!(jacobi.diagonal.v, diagonal)
    preconditioned = VelocityConjugateGradientSolver(toy_mesh_operator!, mesh; preconditioner = jacobi,
                                                     iteration_control = Convergence(; reltol = 1e-12, maxiter = 500))
    xᴾ = VelocityVector(mesh)
    solve!(xᴾ, preconditioned, b, mesh)
    @test Array(interior(xᴾ.u)) ≈ Array(interior(x.u))
    @test Array(interior(xᴾ.v)) ≈ Array(interior(x.v))
    @test preconditioned.iteration <= solver.iteration

    fixed = VelocityConjugateGradientSolver(toy_mesh_operator!, mesh; iteration_control = FixedIterations(4))
    picard = PicardIteration(mesh; iteration_control = FixedIterations(3))
    x³ = VelocityVector(mesh)
    picard_solve!(x³, picard, fixed, toy_update_coefficients!, b, mesh)
    @test fixed.iteration == 4

    allocations = allocations_after_warmup(picard_solve!, x³, picard, fixed, toy_update_coefficients!, b, mesh)
    @info "Triangle mesh Picard(FixedIterations(3)) × CG(FixedIterations(4)) allocations" allocations
    arch isa CPU && @test allocations == 0
end
