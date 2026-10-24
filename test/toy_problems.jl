using KernelAbstractions: @kernel, @index
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Operators: ℑxᶠᵃᵃ, ∂xᶠᶜᶜ

# q = p + Σ ηH (p - p_neighbour): identity plus a weighted graph Laplacian, SPD on periodic grids.
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
