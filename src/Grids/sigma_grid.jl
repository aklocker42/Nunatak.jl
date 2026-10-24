"""
$(TYPEDSIGNATURES)

Return `Nσ + 1` faces of the normalised height above the bed, `ζ = 1 - σ ∈ [0, 1]`,
with `ζ = 0` at the bed and `ζ = 1` at the ice surface. `bed_refinement > 1` clusters
levels towards the bed, where vertical shear and the temperate layer sit.

```jldoctest
julia> using Nunatak

julia> sigma_faces(4; bed_refinement = 1)
5-element Vector{Float64}:
 0.0
 0.25
 0.5
 0.75
 1.0
```
"""
function sigma_faces(Nσ; bed_refinement = 2)
    return [(k / Nσ)^bed_refinement for k in 0:Nσ]
end

"""
$(TYPEDSIGNATURES)

Return a three-dimensional grid with the horizontal coordinates of the two-dimensional
`grid` and `Nσ` levels in the normalised height above the bed `ζ = 1 - σ ∈ [0, 1]`.
Fields on this grid hold column quantities such as the viscosity `η(σ)` and the velocity
`u(σ)`; the physical height is `z = b + ζ H`.

```jldoctest
julia> using Oceananigans, Nunatak

julia> grid = RectilinearGrid(size = (4, 3), x = (0, 4e3), y = (0, 3e3), topology = (Periodic, Bounded, Flat));

julia> σgrid = SigmaGrid(grid; Nσ = 6);

julia> size(σgrid)
(4, 3, 6)
```
"""
function SigmaGrid(grid::RectilinearGrid; Nσ = 10, bed_refinement = 2)
    TX, TY, TZ = topology(grid)
    TZ === Flat || throw(ArgumentError("SigmaGrid needs a horizontal grid with Flat z topology, got $TZ"))

    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    topo = (TX, TY, Bounded)

    horizontal = (x = cpu_face_constructor_x(grid),
                  y = cpu_face_constructor_y(grid))

    horizontal = NamedTuple(name => value for (name, value, T) in zip(keys(horizontal), values(horizontal), (TX, TY)) if T !== Flat)

    sz = pop_flat_elements((Nx, Ny, Nσ), topo)
    halo = pop_flat_elements((Hx, Hy, 1), topo)

    return RectilinearGrid(architecture(grid), eltype(grid);
                           size = sz, halo, topology = topo,
                           z = sigma_faces(Nσ; bed_refinement), horizontal...)
end
