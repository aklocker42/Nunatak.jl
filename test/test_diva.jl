using Nunatak: ice_sheet_velocities, ice_sheet_auxiliary_fields, update_diva_coefficients!, compute_velocity_profile!

# Uniform slab under a uniform driving stress τ with linear sliding: membrane stresses vanish and
# DIVA must reproduce the shallow-ice profile u(ζ) = τ / C + 2A τⁿ H (1 - (1 - ζ)ⁿ⁺¹) / (n + 1).
function slab_profile_error(Nσ; H₀ = 1000, τ₀ = 1e4, A = 1e-21, C = 1e10)
    grid = RectilinearGrid(arch; size = (4, 4), x = (0, 4e3), y = (0, 4e3), topology = (Periodic, Periodic, Flat))
    σgrid = SigmaGrid(grid; Nσ)
    velocities = ice_sheet_velocities(grid, σgrid)
    (; ησ, η̄H, η̄Hᶠᶠ, F₂, βᵘ, βᵛ, βᵇᵘ, βᵇᵛ) = ice_sheet_auxiliary_fields(grid, σgrid)

    H = CenterField(grid)
    set!(H, H₀)
    fill_halo_regions!(H)
    φᵘ, φᵛ = XFaceField(grid), YFaceField(grid)
    set!(φᵘ, 1)
    set!(φᵛ, 1)

    τ = VelocityVector(grid)
    set!(τ.u, τ₀ * 1e3^2)

    rheology = GlenFlowLaw(grid; rate_factor = A, strain_rate_regularization = 1e-16)
    friction = WeertmanFriction(grid; coefficient = C, exponent = 1)
    coefficients = (η̄H, η̄Hᶠᶠ, βᵘ, βᵛ, ησ, F₂, H, rheology, friction, φᵘ, φᵛ, nothing, βᵇᵘ, βᵇᵛ, velocities.uᵇ, velocities.vᵇ)

    solver = VelocityConjugateGradientSolver(ssa_operator!, grid; iteration_control = Convergence(; reltol = 1e-12, maxiter = 100))
    picard = PicardIteration(grid; iteration_control = Convergence(; reltol = 1e-12, maxiter = 300))
    x = VelocityVector(velocities.u, velocities.v)
    iterations = picard_solve!(x, picard, solver, update_diva_coefficients!, τ, coefficients...)
    fill_halo_regions!(ησ)
    compute_velocity_profile!(velocities, βᵇᵘ, βᵇᵛ, F₂, ησ, H)

    n = rheology.exponent
    uᵇ = τ₀ / C
    U = 2A * τ₀^n * H₀ / (n + 1)
    ζ = Array(znodes(σgrid, Center()))
    exact = uᵇ .+ U .* (1 .- (1 .- ζ) .^ (n + 1))
    uσ = Array(interior(velocities.uσ, 1, 1, :))
    ū = Array(interior(velocities.u))
    ūᵉ = uᵇ + 2A * τ₀^n * H₀ / (n + 2)

    return (; iterations,
              profile = maximum(abs, uσ .- exact) / U,
              mean = maximum(abs, ū .- ūᵉ) / (ūᵉ - uᵇ),
              basal = maximum(abs, Array(interior(velocities.uᵇ)) .- uᵇ) / uᵇ,
              v = maximum(abs, Array(interior(velocities.vσ))))
end

@testset "DIVA" begin
    coarse = slab_profile_error(10)
    fine = slab_profile_error(40)
    @info "DIVA slab vs shallow-ice profile" coarse fine
    @test 0 < fine.iterations < 300
    @test fine.profile < 1e-2
    @test fine.mean < 1e-2
    @test fine.basal < 1e-6
    @test fine.v == 0
    @test fine.profile < coarse.profile / 4
end
