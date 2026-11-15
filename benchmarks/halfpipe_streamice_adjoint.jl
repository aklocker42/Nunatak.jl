# Adjoint variant of the MITgcm STREAMICE verification experiment halfpipe_streamice (input_ad): three 0.2-year
# steps with a fixed calving front and cost J = Σ (u² + v² + h²) over full ice cells, with u, v in m/a at cell
# corners. Its gradient check perturbs the basal melt control xx_bdot (m/a) at four cells along the centerline.
# The control has records every two steps on a four-step cycle, so steps 1-3 see 0.5, 1 and 0.5 times record 1.
# STREAMICE solves the velocity at the start of each step, so its cost takes u, v from the start of step 3.
# Nunatak's velocities sit on cell faces and are averaged onto corners; STREAMICE stops at 20 Picard iterations.
using Oceananigans, Nunatak, Enzyme, Printf
using Oceananigans.BoundaryConditions: fill_halo_regions!

include("halfpipe_streamice.jl")

control_weights = (0.5, 1.0, 0.5)
check_cells = ((10, 10), (12, 10), (14, 10), (16, 10))

# From verification/halfpipe_streamice/results/output_adm.txt.
streamice_cost = 5.47764430123086e8
streamice_adjoint_gradient = (-6.9792294581168e2, -1.1933299303989e3, -1.5531439693810e3, -1.7897607038391e3)
streamice_finite_difference = (-6.9781500160694e2, -1.1939452689886e3, -1.5533950144053e3, -1.7896168482304e3)

function halfpipe_adjoint_model(; arch = CPU(), Nσ = 10)
    grid = RectilinearGrid(arch; size = (Nx, Ny), halo = (4, 4), x = (0, Nx * Δx), y = (0, Ny * Δx),
                           topology = (Bounded, Bounded, Flat))
    rheology = GlenFlowLaw(grid; rate_factor = (B^2)^(-n) / year, exponent = n, strain_rate_regularization = 1e-12 / year)
    friction = WeertmanFriction(grid; coefficient = C^2 * year, exponent = 1, velocity_regularization = 1e-12 / year)
    velocity_solver = StressBalanceSolver(grid; linear_iteration_control = Convergence(; reltol = 1e-10, maxiter = 10_000),
                                          picard_iteration_control = Convergence(; reltol = 1e-8, maxiter = 500))
    model = IceSheetModel(grid; Nσ, rheology, friction, velocity_solver,
                          bed = repeat(bed_profile', Nx),
                          stress_balance = DepthIntegrated(shear = DIVAShear(), side_walls = NoSlipWalls()),
                          grounding = SubgridGroundedFraction(), masking = PartialCellFront(grid; passes = 0),
                          constants = IceSheetConstants(grid; ice_density = ρᵢ, ocean_density = ρw, gravitational_acceleration = g),
                          surface_mass_balance = (x, y) -> ifelse(x < Δx, inflow / Δx, zero(x)))
    set!(model, H = initial_thickness)
    return model
end

# Corner velocities in m/a, zero on the walls.
function velocity_cost(model, year)
    u, v, μ = model.velocities.u, model.velocities.v, model.geometry.ice_mask
    Nx, Ny, _ = size(model.grid)
    J = zero(eltype(model.grid))
    for j in 1:Ny, i in 1:Nx
        uᶜ = ifelse(j == 1, zero(J), @inbounds(u[i, j-1, 1] + u[i, j, 1]) / 2) * year
        vᶜ = ifelse(i == 1, zero(J), @inbounds(v[i-1, j, 1] + v[i, j, 1]) / 2) * year
        J += ifelse(@inbounds(μ[i, j, 1]) == 1, uᶜ^2 + vᶜ^2, zero(J))
    end
    return J
end

function thickness_cost(model)
    H, μ = model.geometry.H, model.geometry.ice_mask
    Nx, Ny, _ = size(model.grid)
    J = zero(eltype(model.grid))
    for j in 1:Ny, i in 1:Nx
        J += ifelse(@inbounds(μ[i, j, 1]) == 1, @inbounds(H[i, j, 1])^2, zero(J))
    end
    return J
end

function adjoint_cost(model, xx, weights, Δt, year)
    m = model.basal_melt_rate
    Nx, Ny, _ = size(model.grid)
    J = zero(eltype(xx))
    for (step, w) in enumerate(weights)
        for j in 1:Ny, i in 1:Nx
            @inbounds m[i, j, 1] = w * xx[i, j] / year
        end
        fill_halo_regions!(m)
        step == length(weights) && (J += velocity_cost(model, year))
        time_step!(model, Δt)
    end
    return J + thickness_cost(model)
end

function halfpipe_streamice_adjoint(; arch = CPU(), Nσ = 10, Δt = 0.2year, ϵ = 0.1)
    xx = zeros(Nx, Ny)
    model = halfpipe_adjoint_model(; arch, Nσ)
    dmodel = Enzyme.make_zero(model)
    dxx = zero(xx)
    _, J = autodiff(set_runtime_activity(ReverseWithPrimal), adjoint_cost, Active,
                    Duplicated(model, dmodel), Duplicated(xx, dxx), Const(control_weights), Const(Δt), Const(year))
    @printf("cost  Nunatak %.8e  STREAMICE %.8e  ratio %.4f\n", J, streamice_cost, J / streamice_cost)

    perturbed_cost(δ, i, j) = adjoint_cost(halfpipe_adjoint_model(; arch, Nσ), (x = zeros(Nx, Ny); x[i, j] = δ; x), control_weights, Δt, year)

    rows = NTuple{6, Float64}[]
    for (k, (i, j)) in enumerate(check_cells)
        fd = (perturbed_cost(ϵ, i, j) - perturbed_cost(-ϵ, i, j)) / 2ϵ
        push!(rows, (i, j, dxx[i, j], fd, streamice_adjoint_gradient[k], streamice_finite_difference[k]))
        @printf("(%2d, %2d)  Nunatak ad % .6e  fd % .6e  STREAMICE ad % .6e  fd % .6e  Nunatak/STREAMICE %.4f\n",
                rows[end]..., dxx[i, j] / streamice_adjoint_gradient[k])
        flush(stdout)
    end
    return (; cost = J, rows)
end
