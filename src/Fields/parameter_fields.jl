"""
$(TYPEDSIGNATURES)

Return a two-dimensional `Field` at `(LX, LY, Nothing)` on `grid` set to `value`. A `Field`
passed as `value` is returned unchanged, so parameters always live in fields that an AD tool
can shadow.
"""
function parameter_field(grid, value, LX = Center, LY = Center)
    field = Field{LX, LY, Nothing}(grid)
    set!(field, value)
    fill_halo_regions!(field)
    return field
end

parameter_field(grid, value::Field, LX = Center, LY = Center) = value
parameter_field(grid, ::Nothing, LX = Center, LY = Center) = nothing
