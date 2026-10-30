# Set the resources

```python
--8<-- "examples/03_gw_wannier90_from_scratch.py:resources"
```

`set_resources(builder, step, nodes=, mpi=, threads=, time=, queue=, account=, prepend_text=, custom_scheduler_commands=, npool=)`
changes the job options of one step everywhere it appears in the builder (for example `gw` is in the convergence and in the QP
calculation); the options you do not give are not changed.

| Step | Where |
| --- | --- |
| `gw_scf`, `gw_nscf` | the Quantum ESPRESSO steps of the GW calculations |
| `gw` | p2y and GW |
| `ypp` | `ypp` |
| `wannier_scf`, `wannier_nscf`, `wannier_projwfc`, `wannier_pw2wannier90` | the Quantum ESPRESSO steps of the Wannier workflow |
| `wannier90` | both wannier90 runs |
| `gw2wannier90` | the sorting script |
| `'all'` | every step |

`threads` also sets `OMP_NUM_THREADS`. `npool` is only for the Quantum ESPRESSO steps. The QP splitter has its own resources and
parallelism (`set_qp_subsets(..., resources=..., parallelism='auto')`). To give resources which depend on the values of the GW
parameters during the convergence, use the `resource_rules` of `YamboConvergence` (documentation of `aiida-yambo`, "Resources and
parallelism"): pass them in `overrides={'yambo': {...}}`.
