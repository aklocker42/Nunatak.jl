# Slab under surface cooling, warming and cooling (Kleiner et al., 2015, Exp A):
# compare the basal melt rate with the analytic solution while the basal water layer refreezes.
using Oceananigans, Oceananigans.Units, Nunatak, Printf

year = years = 31556926seconds
ρ, ρw, g, c, k, L, β = 910.0, 1000.0, 9.81, 2009.0, 2.1, 3.34e5, 7.9e-8
T₀, Tᵣ = 273.15, 223.15
H, qᵍ = 1000.0, 0.042
Tc, Tw = T₀ - 30, T₀ - 10
Tₚₘ = T₀ - β * ρ * g * H

# Water-equivalent basal melt rate after the switch to cold surface, starting from the steady warm profile.
function cooling_melt_rate(t; Nmodes = 20_000)
    κ = k / (ρ * c)
    series = sum((-1)^(n + 1) * exp(-κ * (n * π / H)^2 * t) for n in 1:Nmodes)
    ∂T∂z = (Tc - Tₚₘ) / H + 2 * (Tw - Tc) / H * series
    return (qᵍ + k * ∂T∂z) / (ρw * L) * year
end

function surface_temperature_cycle(; Nσ = 100, Δt = 10years, output = "kleiner2015_a.csv")
    grid = RectilinearGrid(CPU(); size = (1, 1), x = (0, 1e3), y = (0, 1e3), topology = (Periodic, Periodic, Flat))
    thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux = qᵍ, surface_temperature = Tc, heat_capacity = c,
                                            thermal_conductivity = k, latent_heat = L, clausius_clapeyron = β,
                                            melting_temperature = T₀, reference_temperature = Tᵣ,
                                            maximum_drainage_rate = 0, grounded_melt_coupling = false)
    model = IceSheetModel(grid; Nσ, bed_refinement = 1, thermodynamics, constants = IceSheetConstants(grid; ice_density = ρ),
                          velocity_solver = nothing, advection = UpwindBiased(order = 1))
    set!(model; H, T = Tc)
    th = model.thermodynamics
    cooling_start = 150_000years
    rows = NTuple{4, Float64}[]

    function record_melt(sim)
        t = time(sim)
        melt = Array(interior(th.state.mᵍ))[1] * year * ρ / ρw
        water = Array(interior(th.state.Hw))[1] * ρ / ρw
        exact = t > cooling_start ? cooling_melt_rate(t - cooling_start) : NaN
        push!(rows, (t / year, melt, water, exact))
        return nothing
    end

    simulation = Simulation(model; Δt)
    add_callback!(simulation, record_melt, IterationInterval(1))

    for (Tₛ, stop_time) in ((Tc, 100_000years), (Tw, cooling_start), (Tc, 300_000years))
        set!(th.surface_temperature, Tₛ)
        simulation.stop_time = stop_time
        run!(simulation)
    end
    open(output, "w") do io
        println(io, "t_years,melt_we_m_per_a,water_we_m,melt_exact_we_m_per_a")
        foreach(r -> println(io, join(r, ",")), rows)
    end
    return rows
end

rows = surface_temperature_cycle()
cooling = filter(r -> r[1] > 150_000 && r[3] > 0, rows)

function sign_change(rows, column)
    for (r⁻, r⁺) in zip(rows, rows[2:end])
        r⁻[column] > 0 ≥ r⁺[column] && return r⁻[1] + (r⁺[1] - r⁻[1]) * r⁻[column] / (r⁻[column] - r⁺[column]) - 150_000
    end
    return NaN
end

@printf("warm steady melt %.4e, analytic %.4e m/a w.e.\n", rows[findfirst(r -> r[1] ≥ 150_000, rows)][2],
        (qᵍ - k * (Tₚₘ - Tw) / H) / (ρw * L) * year)
@printf("melt to freeze after %.1f a, analytic %.1f a\n", sign_change(cooling, 2), sign_change(cooling, 4))
@printf("max |melt - analytic| with basal water %.3e m/a w.e.\n", maximum(r -> abs(r[2] - r[4]), cooling))
@printf("final melt %.4e, analytic %.4e m/a w.e.\n", cooling[end][2], cooling[end][4])
