# EISMINT II Exp A (Payne et al., 2000, J. Glaciol. 46(153)): thermomechanically coupled ice cap grown from zero ice
# on a flat bed under radially symmetric accumulation and surface temperature, compared with Table 4 after 200 kyr.
using Oceananigans, Nunatak, Printf, DelimitedFiles

spy = 31556926
ρ, g, c, k, L, G = 910.0, 9.81, 2009.0, 2.1, 3.35e5, 0.042
T₀, βz = 273.15, 8.66e-4
Mₘₐₓ, Sᵦ, Rₑₗ = 0.5, 1e-2, 450.0
Tₘᵢₙ, Sₜ = 238.15, 1.67e-2
x₀ = y₀ = 750e3

distance(x, y) = sqrt((x - x₀)^2 + (y - y₀)^2) / 1e3
accumulation(x, y) = min(Mₘₐₓ, Sᵦ * (Rₑₗ - distance(x, y))) / spy
surface_air_temperature(x, y) = Tₘᵢₙ + Sₜ * distance(x, y)

function eismint2_a(; arch = CPU(), stress_balance = ShallowIce(), Nσ = 20, Δt = 5spy, stop_time = 200_000spy,
                      report_interval = 1000spy, output = "eismint2_a")
    Δx = 25e3
    grid = RectilinearGrid(arch; size = (61, 61), halo = (4, 4), x = (-Δx / 2, 1500e3 + Δx / 2), y = (-Δx / 2, 1500e3 + Δx / 2),
                           topology = (Bounded, Bounded, Flat))

    # Step in A at 263.15 K and no water softening, as specified.
    rate_factor = ArrheniusRateFactor(grid; gas_constant = 8.314, transition_width = 1e-3, water_softening_coefficient = 0)
    rheology = GlenFlowLaw(grid; rate_factor)
    friction = WeertmanFriction(grid; coefficient = 1e18, exponent = 1)
    thermodynamics = EnthalpyThermodynamics(grid; geothermal_flux = G, surface_temperature = surface_air_temperature,
                                            heat_capacity = c, thermal_conductivity = k, latent_heat = L,
                                            clausius_clapeyron = βz / (ρ * g), melting_temperature = T₀,
                                            grounded_melt_coupling = false)
    model = IceSheetModel(grid; Nσ, rheology, friction, thermodynamics, stress_balance,
                          constants = IceSheetConstants(grid; ice_density = ρ, gravitational_acceleration = g),
                          surface_mass_balance = accumulation)

    th = model.thermodynamics
    i₀ = j₀ = 31
    Ωᶜ = Δx^2
    rows = NTuple{6, Float64}[]
    wall = time_ns()
    Nt = round(Int, stop_time / Δt)
    Nreport = round(Int, report_interval / Δt)

    function diagnostics()
        H = Array(interior(model.geometry.H, :, :, 1))
        E₁ = Array(interior(th.state.E, :, :, 1))
        qᵍ = Array(interior(th.state.qᵍ, :, :, 1))
        Dᵇ = Array(interior(th.state.Dᵇ, :, :, 1))
        rᵇ = Array(interior(th.state.rᵇ, :, :, 1))
        Tₚₘ = T₀ .- βz .* H
        Tᵇ = min.(th.parameters.reference_temperature .+ (E₁ .+ qᵍ ./ Dᵇ) ./ c, Tₚₘ)
        ice = H .> 1
        volume = sum(H[ice]) * Ωᶜ
        area = count(ice) * Ωᶜ
        melt = count(ice .& (rᵇ .≥ 0.5)) / max(count(ice), 1)
        return (; H, Tᵇ, rᵇ, volume, area, melt, divide_thickness = H[i₀, j₀], divide_temperature = Tᵇ[i₀, j₀])
    end

    for n in 1:Nt
        time_step!(model, Δt)
        if n % Nreport == 0 || n == Nt
            d = diagnostics()
            push!(rows, (model.clock.time / spy, d.volume / 1e15, d.area / 1e12, d.melt, d.divide_thickness, d.divide_temperature))
            @printf("t = %7.0f a  V = %.4f 10⁶ km³  A = %.4f 10⁶ km²  melt = %.3f  H₀ = %8.2f m  Tᵇ₀ = %7.3f K  wall %.0f s\n",
                    rows[end]..., (time_ns() - wall) / 1e9)
            flush(stdout)
            isfinite(d.divide_thickness) || break
        end
    end

    d = diagnostics()
    writedlm(output * "_timeseries.csv", [["t_years" "volume_1e6km3" "area_1e6km2" "melt_fraction" "divide_thickness_m" "divide_basal_T_K"];
                                          reduce(vcat, [collect(r)' for r in rows])], ',')
    writedlm(output * "_thickness.csv", d.H, ',')
    writedlm(output * "_basal_temperature.csv", d.Tᵇ, ',')
    writedlm(output * "_basal_ramp.csv", d.rᵇ, ',')
    return rows
end
