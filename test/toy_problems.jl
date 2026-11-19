using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: ℑxᶠᵃᵃ, ∂xᶠᶜᶜ

# q = p + Σ ηH (p - p_neighbor): identity plus a weighted graph Laplacian, SPD on periodic grids.
function toy_operator!(q, p, ηH)
    grid = ηH.grid
    launch!(architecture(grid), grid, :xy, _toy_operator!, q.u, q.v, p.u, p.v, ηH)
    return nothing
end

@kernel function _toy_operator!(qu, qv, pu, pv, ηH)
    i, j = @index(Global, NTuple)
    @inbounds begin
        qu[i, j, 1] = pu[i, j, 1] + ηH[i, j, 1]   * (pu[i, j, 1] - pu[i+1, j, 1]) +
                                    ηH[i-1, j, 1] * (pu[i, j, 1] - pu[i-1, j, 1])
        qv[i, j, 1] = pv[i, j, 1] + ηH[i, j, 1]   * (pv[i, j, 1] - pv[i, j+1, 1]) +
                                    ηH[i, j-1, 1] * (pv[i, j, 1] - pv[i, j-1, 1])
    end
end

toy_update_coefficients!(x, ηH) = nothing

function toy_grid(arch = CPU(); N = 8)
    return RectilinearGrid(arch; size = (N, N), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat))
end

function toy_viscosity(grid)
    ηH = Field{Center, Center, Nothing}(grid)
    set!(ηH, (x, y) -> 1 + sin(2π * x)^2 + cos(2π * y)^2 / 2)
    fill_halo_regions!(ηH)
    return ηH
end

function toy_right_hand_side(grid)
    b = VelocityVector(grid)
    set!(b.u, (x, y) -> sin(2π * x) * cos(4π * y))
    set!(b.v, (x, y) -> cos(2π * x) + x * y)
    return b
end

# Toy driving stress τ = ρg ℑx(H) ∂x s for the Enzyme kernel gate.
function toy_driving_stress!(τ, H, s)
    grid = H.grid
    launch!(architecture(grid), grid, :xy, _toy_driving_stress!, τ, H, s, grid)
    return nothing
end

@kernel function _toy_driving_stress!(τ, H, s, grid)
    i, j = @index(Global, NTuple)
    @inbounds τ[i, j, 1] = 910 * 9.81 * ℑxᶠᵃᵃ(i, j, 1, grid, H) * ∂xᶠᶜᶜ(i, j, 1, grid, s)
end

# Square [0, L]² split into N × N cells of two triangles each; odd cells are listed clockwise on purpose.
function split_square_mesh(N, L; arch = CPU(), kwargs...)
    node(i, j) = i + (N + 1) * (j - 1)
    x = [L * (i - 1) / N for i in 1:N+1, j in 1:N+1]
    y = [L * (j - 1) / N for i in 1:N+1, j in 1:N+1]
    elements = zeros(Int, 3, 2N^2)
    for j in 1:N, i in 1:N
        c = i + N * (j - 1)
        lower = (node(i, j), node(i + 1, j), node(i + 1, j + 1))
        upper = (node(i, j), node(i + 1, j + 1), node(i, j + 1))
        elements[:, 2c - 1] .= isodd(c) ? reverse(lower) : lower
        elements[:, 2c]     .= isodd(c) ? reverse(upper) : upper
    end
    return TriangleMesh(arch; x = vec(x), y = vec(y), elements, kwargs...)
end

# q = p + Σₑ Aₑ (∇φⱼ ⋅ ∇φᵢ) pⱼ: identity plus a P1 stiffness matrix, SPD, assembled by node gather.
function toy_mesh_operator!(q, p, mesh)
    launch!(architecture(mesh), mesh, length(mesh.x), _toy_mesh_operator!, q.u, q.v, p.u, p.v, mesh)
    return nothing
end

@kernel function _toy_mesh_operator!(qu, qv, pu, pv, mesh)
    n = @index(Global, Linear)
    @inbounds begin
        su = pu.data[n]
        sv = pv.data[n]
        for position in mesh.node_element_offsets[n]:mesh.node_element_offsets[n+1]-1
            e = mesh.node_elements[position]
            i = mesh.node_local_index[position]
            for j in 1:3
                m = mesh.elements[j, e]
                w = mesh.area[e] * (mesh.∂φ∂x[j, e] * mesh.∂φ∂x[i, e] + mesh.∂φ∂y[j, e] * mesh.∂φ∂y[i, e])
                su += w * pu.data[m]
                sv += w * pv.data[m]
            end
        end
        qu.data[n] = su
        qv.data[n] = sv
    end
end

toy_mesh_mass_operator!(q, p, mesh) = (q.u.data .= mesh.node_area .* p.u.data; q.v.data .= mesh.node_area .* p.v.data; nothing)
