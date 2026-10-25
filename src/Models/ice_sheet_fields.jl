"""
$(TYPEDSIGNATURES)

Depth-averaged velocities `(u, v)` on the C-grid, basal velocities `(uᵇ, vᵇ)`, and the
vertical structure `(uσ, vσ)` on the σ grid. The DIVA reconstruction fills `uᵇ`, `vᵇ`, `uσ`, `vσ`;
under SSA they equal the depth average.
"""
function ice_sheet_velocities(grid, σgrid)
    u  = Field{Face, Center, Nothing}(grid)
    v  = Field{Center, Face, Nothing}(grid)
    uᵇ = Field{Face, Center, Nothing}(grid)
    vᵇ = Field{Center, Face, Nothing}(grid)
    uσ = Field{Face, Center, Center}(σgrid)
    vσ = Field{Center, Face, Center}(σgrid)
    return (; u, v, uᵇ, vᵇ, uσ, vσ)
end

"""
$(TYPEDSIGNATURES)

Ice thickness `H`, surface elevation `s`, grounded fraction `φ`, effective pressure `N` and ice mask at
`(Center, Center)`, and the grounded fractions `φᵘ`, `φᵛ` at the velocity points.
"""
function ice_sheet_geometry(grid)
    H = Field{Center, Center, Nothing}(grid)
    s = Field{Center, Center, Nothing}(grid)
    φ = Field{Center, Center, Nothing}(grid)
    φᵘ = Field{Face, Center, Nothing}(grid)
    φᵛ = Field{Center, Face, Nothing}(grid)
    N = Field{Center, Center, Nothing}(grid)
    ice_mask = Field{Center, Center, Nothing}(grid)
    return (; H, s, φ, φᵘ, φᵛ, N, ice_mask)
end

"""
$(TYPEDSIGNATURES)

Diagnostic fields of the stress balance: viscosity on the σ grid `ησ`, depth-integrated
viscosity `η̄H` at cell centres and `η̄Hᶠᶠ` at corners, the DIVA integral `F₂`, the drag
coefficients `βᵘ`, `βᵛ` of the depth-averaged operator, the friction-law drag at the bed `βᵇᵘ`, `βᵇᵛ`
(DIVA only; under SSA `βᵘ`, `βᵛ` are the bed drag) and driving stresses `τᵘ`, `τᵛ`.
"""
function ice_sheet_auxiliary_fields(grid, σgrid)
    ησ    = Field{Center, Center, Center}(σgrid)
    η̄H    = Field{Center, Center, Nothing}(grid)
    η̄Hᶠᶠ  = Field{Face, Face, Nothing}(grid)
    F₂    = Field{Center, Center, Nothing}(grid)
    βᵘ    = Field{Face, Center, Nothing}(grid)
    βᵛ    = Field{Center, Face, Nothing}(grid)
    βᵇᵘ   = Field{Face, Center, Nothing}(grid)
    βᵇᵛ   = Field{Center, Face, Nothing}(grid)
    τᵘ    = Field{Face, Center, Nothing}(grid)
    τᵛ    = Field{Center, Face, Nothing}(grid)
    return (; ησ, η̄H, η̄Hᶠᶠ, F₂, βᵘ, βᵛ, βᵇᵘ, βᵇᵛ, τᵘ, τᵛ)
end
