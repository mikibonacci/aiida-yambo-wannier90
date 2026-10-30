# Set the inputs

## The builder

```python
from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

builder = YamboWannier90WorkChain.get_builder_from_protocol(
    codes, structure, protocol='moderate', pseudo_family='PseudoDojo/0.4/PBE/SR/standard/upf',
    wannier_projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,   # or ANALYTIC, SCDM, RANDOM
    electronic_type=ElectronicType.METAL,                                 # or INSULATOR
    overrides={'wannier90': {...}, 'yambo': {...}, 'wannier90_qp': {...}, 'yambo_qp': {...}, 'ypp': {...}},
)
```

- `protocol` (`fast`, `moderate`, `precise`) goes to the GW convergence, the QP calculation and both Wannier90 stages.
- `overrides` has one key for each stage; its value is merged in the inputs of that stage (nested like the inputs).
- `electronic_type` decides the disentanglement of the Wannier functions: metals (the default) disentangle with the frozen window,
  insulators do not. **Choose it for your system.**
- The projection, disentanglement and frozen types are the same for the two Wannier90 runs.
- `codes` is a dictionary by label, pk or node; it is not modified.

## Meshes, bands and Wannier functions

- `set_wannier_mesh(builder, [n1, n2, n3])`: the coarse mesh of both Wannier90 runs and of the nscf calculation.
- `set_gw_mesh(builder, mesh, force=False)`: the GW mesh, to skip its convergence (`force`: Wannier90 uses it too).
- `set_wannier_bands(builder, num_bands, num_wann=None, exclude_bands=None)`: `num_bands` is the number of bands of the nscf
  calculation (including the excluded ones), as in `aiida-wannier90-workflows`.
- `set_projections(builder, ['Si:sp3'])`: explicit analytic projections, in both runs.

The mesh is the one of the cell of the calculations: if seekpath runs, the primitive cell it finds.

## The QP calculation

`set_qp_subsets(builder, qp_per_subset=50, parallel_runs=4, **options)` sets how the QP splitter of `YamboWorkflow` cuts the
calculation: `parallelism='auto'`, `resources`, `max_wallclock_seconds`, `queue_name` and the other keys of `QP_subsets_dict` (see the
documentation of `aiida-yambo`, "QP splitter"; unknown keys raise an error). The quasiparticles themselves are chosen by the
workchain: the bands of the Wannier functions at the coarse k-points, so `boundaries`, `range_QP` and `explicit` are refused.

## See what will run

`print_summary(builder)` prints the stages, the band path, the Wannier meshes, bands and projections, the QP splitter and the
resources of each step.
