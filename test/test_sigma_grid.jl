using Nunatak: compute_column_integrals!, layer_thickness

@testset "σ grid and column integrals" begin
    grid = RectilinearGrid(arch; size = (4, 3), x = (0, 1), y = (0, 1), topology = (Periodic, Bounded, Flat))
    σgrid = SigmaGrid(grid; Nσ = 7, bed_refinement = 2)

    @test size(σgrid) == (4, 3, 7)
    @test sum(layer_thickness(k, on_architecture(CPU(), σgrid)) for k in 1:7) ≈ 1

    η = 3e13
    ησ = CenterField(σgrid)
    set!(ησ, η)
    H = Field{Center, Center, Nothing}(grid)
    set!(H, (x, y) -> 1000 + 500x)
    η̄H = Field{Center, Center, Nothing}(grid)
    F₂ = Field{Center, Center, Nothing}(grid)

    compute_column_integrals!(η̄H, F₂, ησ, H)

    @test Array(interior(η̄H)) ≈ η .* Array(interior(H))
    @test Array(interior(F₂)) ≈ Array(interior(H)) ./ (3η)
end
