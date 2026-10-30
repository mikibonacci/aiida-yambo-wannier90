# How the workchain works

`YamboWannier90WorkChain` runs these stages, in this order. A stage which has no inputs is skipped, so the workchain can start from
the middle ([how-to](../how-to/stages.md)).

```text
seekpath -> yambo (GW convergence) -> kmesh -> yambo_commensurate -> yambo_qp
         -> wannier90_pp -> ypp -> wannier90 -> gw2wannier90 -> wannier90_qp
```

| Stage | What it does |
| --- | --- |
| `seekpath` | finds the primitive cell and the band path, unless you give `kpoint_path` or `bands_kpoints` |
| `yambo` | `YamboConvergence`: converges the GW parameters, and the k-mesh, on a gap |
| `kmesh` | finds a GW mesh and a (coarse) Wannier90 mesh which are commensurate: each coarse k-point is in the GW mesh |
| `yambo_commensurate` | repeats the converged GW calculation on the commensurate mesh, if it is not the converged one |
| `yambo_qp` | `YamboWorkflow` with the QP splitter: the QP of the bands of the Wannier functions, at the coarse k-points |
| `wannier90_pp` | `wannier90 -pp`: writes the `nnkp` file (the neighbours of the k-points) |
| `ypp` | writes `unsorted.eig`: the GW energies, for the QP k-points and bands |
| `wannier90` | the DFT Wannierization and the interpolated DFT bands |
| `gw2wannier90` | puts the GW energies in the `eig` file read by wannier90 |
| `wannier90_qp` | wannier90 again, with the GW energies and the same Wannier functions: the interpolated GW bands |

The outputs are `band_structures.wannier90` (DFT) and `band_structures.wannier90_qp` (GW), a namespace for every stage, and
`primitive_structure` / `seekpath_parameters` if seekpath ran.

## The two meshes

Wannier90 needs the GW energies on its (coarse) k-mesh, so every coarse k-point must be one of the k-points of the GW calculation.
Given the converged GW mesh and the Wannier90 mesh you gave, the workchain looks for the closest commensurate pair (or uses the GW
mesh for Wannier90 if `kpoints_force_gw` is true). `kmapper` maps each coarse point to the (symmetry-reduced) GW list and sets the
quasiparticles to compute. If you start after the convergence (`GW_mesh` or a `parent_folder`), the commensurability is checked in
`setup` and the workchain stops with the exit code 401 if the meshes do not match.

## The two Wannier90 runs

The GW run uses the matrices of the first one (`amn`, `mmn`), so both must have the same number of bands and Wannier functions, the
same projections and the same mesh. The builder and the setters (`set_wannier_bands`, `set_projections`, `set_wannier_mesh`) keep them
equal, and the workchain refuses inputs where they differ.

## The band path

One of `kpoint_path` (high-symmetry points, wannier90 interpolates along them), `bands_kpoints` (an explicit list of points, used as it
is) or `bands_kpoints_distance` (seekpath), or none (seekpath). More than one is an error. See [the band path](../how-to/kpath.md).
