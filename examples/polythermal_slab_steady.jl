# Steady polythermal slab with prescribed strain heating (Kleiner et al., 2015, Exp B):
# compare the cold-temperate transition surface with the analytic solution.
using Oceananigans, Nunatak, Printf

module SteadyPolythermalSlab

const spy = 31556926
const ρ, g, c, k, L = 910.0, 9.81, 2009.0, 2.1, 3.35e5
const Tᵣ, T₀ = 223.15, 273.15
const H, A, γ = 200.0, 5.3e-24, deg2rad(4)
const a = 0.2 / spy
const Tₛ = T₀ - 3

const D = k / (c * ρ)
const M = H * a
const K = 2A * (ρ * g * sin(γ))^4 * H^6 / ρ
const Eₛ = c * (Tₛ - Tᵣ)
const Eₚₘ = c * (T₀ - Tᵣ)

# Ep′ = Σ bⱼ ζʲ solves D Ep″ + M Ep′ = -K (1 - ζ)⁴.
function particular_coefficients()
    f = [-K * binomial(4, j) * (-1)^j for j in 0:4]
    b = zeros(5)
    b[5] = f[5] / M
    for j in 3:-1:0
        b[j+1] = (f[j+1] - D * (j + 1) * b[j+2]) / M
    end
    return b
end

const b = particular_coefficients()
gradient_particular(ζ) = sum(b[j+1] * ζ^j for j in 0:4)
particular(ζ) = sum(b[j] / j * ζ^j for j in 1:5)

function constants(ζₘ)
    c₁ = D / M * exp(M * ζₘ / D) * gradient_particular(ζₘ)
    c₂ = Eₛ - c₁ * exp(-M / D) - particular(1)
    return c₁, c₂
end

function transition_mismatch(ζₘ)
    c₁, c₂ = constants(ζₘ)
    return c₁ * exp(-M * ζₘ / D) + c₂ + particular(ζₘ) - Eₚₘ
end

function transition_height(; lo = 1e-6, hi = 0.5)
    flo = transition_mismatch(lo)
    for _ in 1:200
        mid = (lo + hi) / 2
        fmid = transition_mismatch(mid)
        if sign(fmid) == sign(flo)
            lo, flo = mid, fmid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

const ζₘ = transition_height()

function enthalpy(ζ)
    c₁, c₂ = constants(ζₘ)
    cold = c₁ * exp(-M * ζ / D) + c₂ + particular(ζ)
    temperate = Eₚₘ + K / (5M) * ((1 - ζ)^5 - (1 - ζₘ)^5)
    return ifelse(ζ < ζₘ, temperate, cold)
end

end

using .SteadyPolythermalSlab: spy, ρ, g, c, k, L, Tᵣ, T₀, H, A, γ, a, Tₛ, Eₚₘ, ζₘ

function steady_polythermal_slab(Nσ, ratio; Δt = 10spy, stop_time = 30_000spy)
    grid = RectilinearGrid(CPU(); size = (1, 1), x = (0, 1e3), y = (0, 1e3), topology = (Periodic, Periodic, Flat))
    thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux = 0, surface_temperature = Tₛ, heat_capacity = c,
                                            thermal_conductivity = k, latent_heat = L, clausius_clapeyron = 0,
                                            melting_temperature = T₀, reference_temperature = Tᵣ,
                                            temperate_diffusivity_ratio = ratio, maximum_drainage_rate = 0,
                                            grounded_melt_coupling = false)
    model = IceSheetModel(grid; Nσ, bed_refinement = 1, thermodynamics, constants = IceSheetConstants(grid; ice_density = ρ),
                          velocity_solver = nothing, advection = UpwindBiased(order = 1),
                          surface_mass_balance = a, basal_melt_rate = a)
    set!(model; H, T = T₀ - 1.5)
    th = model.thermodynamics
    ζᶠ = collect(znodes(model.σgrid, Face()))
    ζᶜ = collect(znodes(model.σgrid, Center()))

    # Layer averages of the strain heating of a parallel-sided slab with Glen n = 3.
    K′ = 2A * (ρ * g * sin(γ))^4 * H^4
    interior(th.state.Φ)[1, 1, :] .= [K′ * ((1 - ζᶠ[n])^5 - (1 - ζᶠ[n+1])^5) / (5 * (ζᶠ[n+1] - ζᶠ[n])) for n in 1:Nσ]

    for _ in 1:round(Int, stop_time / Δt)
        Nunatak.advance_enthalpy!(model, th, Δt)
    end

    E = Array(interior(th.state.E, 1, 1, :))
    n = findfirst(>(0), Eₚₘ .- E)
    ζ = n == 1 ? zero(H) : ζᶜ[n-1] + (ζᶜ[n] - ζᶜ[n-1]) * (E[n-1] - Eₚₘ) / (E[n-1] - E[n])
    error = maximum(abs, E .- SteadyPolythermalSlab.enthalpy.(ζᶜ)) / c
    return (; Nσ, ratio, height = ζ * H, height_error = (ζ - ζₘ) * H, error)
end

@printf("analytic transition height %.3f m\n", ζₘ * H)
@printf("%5s %7s %12s %12s %14s\n", "Nσ", "ratio", "height (m)", "error (m)", "max|ΔE|/c (K)")
for Nσ in (20, 40, 100, 400), ratio in (1e-1, 1e-2, 1e-3, 1e-4)
    r = steady_polythermal_slab(Nσ, ratio)
    @printf("%5d %7.0e %12.3f %12.3f %14.4e\n", r.Nσ, r.ratio, r.height, r.height_error, r.error)
end
