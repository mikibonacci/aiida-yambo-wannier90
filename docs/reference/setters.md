# Setters

All in `aiida_yambo_wannier90.utils.setters`; they act on the builder.

| Function | What it sets |
| --- | --- |
| `set_bands_kpath(builder, path=, point_coords=, kpoints=, explicit_kpoints=, distance=)` | the band path (one source) |
| `set_wannier_mesh(builder, mesh)` | the coarse mesh: both wannier90 runs and the nscf calculation |
| `set_gw_mesh(builder, mesh, force=False)` | the GW mesh (`GW_mesh`, `kpoints_force_gw`) |
| `set_wannier_bands(builder, num_bands, num_wann=None, exclude_bands=None)` | bands and Wannier functions, in both runs |
| `set_projections(builder, projections)` | analytic projections, in both runs |
| `sync_wannier90_qp(builder)` | copies `num_wann`, `num_bands`, `exclude_bands`, projections to the QP run |
| `set_qp_subsets(builder, qp_per_subset, parallel_runs, **options)` | the QP splitter of the QP calculation |
| `set_resources(builder, step, ...)` | job options of a step (`STEPS`) |
| `start_from(builder, stage, **data)` | remove the earlier stages and give what they produce (`START_INPUTS`) |
| `print_summary(builder)` | print what will run |

`aiida_yambo_wannier90.calculations.functions.kpath.kpoints_from_path(structure, path, point_coords)` builds the labelled
`KpointsData` of a path.

## Inputs of the workchain

| Input | Meaning |
| --- | --- |
| `structure` | the structure |
| `kpoint_path` | labelled high-symmetry points (wannier90 interpolates along them) |
| `bands_kpoints` | labelled explicit list of k-points |
| `bands_kpoints_distance` | seekpath reference distance |
| `GW_mesh`, `kpoints_force_gw` | the GW mesh, to skip the convergence; use it also for wannier90 |
| `yambo` | `YamboConvergence` inputs |
| `yambo_qp` | `YamboWorkflow` inputs (`scf`, `nscf`, `qp`, `QP_subsets_dict`, `parent_folder`...) |
| `ypp`, `wannier90`, `gw2wannier90`, `wannier90_qp` | the inputs of each stage |
| `clean_workdir` | clean the remote folders at the end |

Only one of `kpoint_path`, `bands_kpoints`, `bands_kpoints_distance` can be given.
