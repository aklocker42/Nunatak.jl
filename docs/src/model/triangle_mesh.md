# Triangle meshes

Besides Oceananigans grids, the SSA velocity solve runs on unstructured meshes of linear (P1) triangles, which can
follow a coastline and refine where the flow needs it. The formulation is the one of ISSM and DJUICE; the kernels
run on CPUs and GPUs and the solve is differentiable with Enzyme, like the structured one. Thickness transport,
grounding-line migration, DIVA and thermodynamics on triangles are not available yet.

## Building a mesh

[`TriangleMesh`](@ref) takes node coordinates and a `3 × Nelements` matrix of node indices, for example from
Triangulate.jl. Elements are reoriented counter-clockwise, and degenerate elements are rejected. Each velocity
component can be constrained to zero on a set of nodes (by default every boundary node), and boundary edges listed
in `front_segments` carry the ice-front stress. Values live on nodes ([`NodeField`](@ref)) or elements
([`ElementField`](@ref)); `set!` accepts a number, a function of ``(x, y)`` or a vector.

## Discretisation

With P1 basis functions ``\phi_i`` the gradients ``b_i = \partial \phi_i / \partial x`` and
``c_i = \partial \phi_i / \partial y`` are constant on each element ``e`` of area ``A_e``, so the element
integrals are exact in closed form. With the depth-integrated viscosity ``\overline{\eta H}_e = \eta_e \bar H_e``,
``\bar H_e`` the mean of the three nodal thicknesses, the viscous block coupling local nodes ``i`` and ``j`` is

```math
\overline{\eta H}_e A_e
\begin{pmatrix} 4 b_i b_j + c_i c_j & 2 b_i c_j + c_i b_j \\ 2 c_i b_j + b_i c_j & 4 c_i c_j + b_i b_j \end{pmatrix} .
```

The element viscosity follows Glen's law from the element strain rates, with the same regularisation as on the
grid. Basal drag is lumped on the nodes, ``A_n \beta_n \mathbf{u}_n`` with ``A_n = \sum_{e \ni n} A_e / 3``, as the
C-grid does on its velocity points; ISSM integrates it with a consistent mass matrix, which differs at
``O(h^2)``. The driving stress is ``-\rho_i g \, \nabla s \int_e H \phi_n \, dA`` with the exact integral
``A_e (2 H_n + H_j + H_k)/12``. On a front segment the load adds
``\int \left(\tfrac{1}{2} \rho_i g H^2 - \tfrac{1}{2} \rho_w g D^2\right) \phi_n \hat{\mathbf{n}} \, dl``, ``D`` the
submerged depth, integrated with two Gauss points.

The operator is applied by a kernel over nodes that gathers the contributions of the elements around each node,
so it needs no atomic operations. Constrained rows are the identity and constrained values enter the other rows
as zero, so the operator stays symmetric and the existing conjugate-gradient solver, Jacobi preconditioner and
Picard iteration are reused unchanged. Only zero constraints are supported so far.

## Solving

`solve_stress_balance!(velocity, solver, geometry, auxiliary, rheology, friction, constants)` updates the
flotation surface and grounded mask, computes the driving stress and runs the Picard iteration;
`StressBalanceSolver(mesh)` selects the triangle operator. The example
[An ice shelf in a bay on a triangle mesh](../literated/triangle_ice_shelf.md) walks through a complete solve,
and the [validation page](../validation.md) compares the velocities with DJUICE on the same meshes.
