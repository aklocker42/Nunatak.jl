# Isothermal Halfar (1983) similarity dome, Bueler et al. (2005, J. Glaciol. 51(173)) Test B: checks the
# chosen stress balance with a no-slip bed against the analytic shallow-ice solution, without thermodynamics.
using Oceananigans, Oceananigans.Units, Nunatak, Printf, DelimitedFiles

year = years = 31556926seconds
ρ, g, n = 910.0, 9.81, 3
H₀, R₀ = 3600.0, 750kilometers
A = 1e-16 / year
Γ = 2A * (ρ * g)^n / (n + 2)
α, β = 1 / 9, 1 / 18
t₀ = β / Γ * ((2n + 1) / (n + 1))^n * R₀^(n + 1) / H₀^(2n + 1)

function halfar_thickness(x, y, t)
    r = sqrt(x^2 + y^2)
    s = (t₀ / t)^β * r / R₀
    return H₀ * (t₀ / t)^α * max(0, 1 - s^((n + 1) / n))^(n / (2n + 1))
end

function halfar(; arch = CPU(), stress_balance = ShallowIce(), Nσ = 10, Δx = 25kilometers, Δt = 1year, stop_time = 25_000years,
                  report_interval = 1000years, output = "halfar")
    L = 1200kilometers
    Nx = round(Int, 2L / Δx) + 1
    grid = RectilinearGrid(arch; size = (Nx, Nx), halo = (4, 4), x = (-L - Δx / 2, L + Δx / 2), y = (-L - Δx / 2, L + Δx / 2),
                           topology = (Bounded, Bounded, Flat))

    rheology = GlenFlowLaw(grid; rate_factor = A, exponent = n)
    friction = WeertmanFriction(grid; coefficient = 1e18, exponent = 1)
    model = IceSheetModel(grid; Nσ, rheology, friction, stress_balance,
                          constants = IceSheetConstants(grid; ice_density = ρ, gravitational_acceleration = g))
    set!(model, H = (x, y) -> halfar_thickness(x, y, t₀))

    simulation = Simulation(model; Δt, stop_time)

    x = xnodes(grid, Center())
    y = ynodes(grid, Center())
    i₀ = j₀ = (Nx + 1) ÷ 2
    rows = NTuple{7, Float64}[]

    function diagnostics()
        t = t₀ + model.clock.time
        H = Array(interior(model.geometry.H, :, :, 1))
        Hₑ = [halfar_thickness(xᵢ, yⱼ, t) for xᵢ in x, yⱼ in y]
        e = H .- Hₑ
        ice = (H .> 1) .| (Hₑ .> 1)
        Vₑ = sum(Hₑ)
        return (; t, H, Hₑ, divide_error = e[i₀, j₀], max_error = maximum(abs, e), mean_error = sum(abs, e[ice]) / count(ice),
                  volume_error = (sum(H) - Vₑ) / Vₑ, divide_thickness = H[i₀, j₀])
    end

    function progress(sim)
        d = diagnostics()
        push!(rows, (d.t / year, time(sim) / year, d.divide_thickness, d.divide_error, d.max_error, d.mean_error, d.volume_error))
        @info @sprintf("t - t₀ = %6.0f a, H₀ = %8.2f m, ΔH₀ = %7.2f m, max|ΔH| = %7.2f m, mean|ΔH| = %6.2f m, ΔV/V = %+.4f, wall time: %s",
                       rows[end][2:end]..., prettytime(sim.run_wall_time))
        return nothing
    end

    add_callback!(simulation, progress, TimeInterval(report_interval))
    run!(simulation)

    d = diagnostics()
    writedlm(output * "_timeseries.csv", [["t_years" "t_minus_t0_years" "divide_thickness_m" "divide_error_m" "max_abs_error_m" "mean_abs_error_m" "relative_volume_error"];
                                          reduce(vcat, [collect(r)' for r in rows])], ',')
    writedlm(output * "_thickness.csv", d.H, ',')
    writedlm(output * "_exact_thickness.csv", d.Hₑ, ',')
    return rows
end
