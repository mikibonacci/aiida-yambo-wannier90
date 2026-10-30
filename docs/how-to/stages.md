# Start from the middle

The stages run in this order: `yambo` (GW convergence) → `yambo_qp` → `ypp` → `wannier90` → `gw2wannier90` → `wannier90_qp`.
`start_from(builder, stage, **data)` removes the earlier stages and sets the data they would have produced.

```python
--8<-- "examples/04_from_existing_gw.py:start_from"
```

| Stage | What you give |
| --- | --- |
| `yambo_qp` | `parent_folder` (the remote folder of a converged GW calculation), or `gw_mesh` to run the QP calculation on that mesh without the convergence |
| `ypp` | `parent_folder` (the QP calculation) and `qp_db` (the merged `ndb.QP`) |
| `wannier90` | `unsorted_eig` and `nnkp` |
| `gw2wannier90` | `parent_folder` (the wannier90 run), `unsorted_eig`, `nnkp` |
| `wannier90_qp` | `remote_input_folder` (the `gw2wannier90` calculation) |

A missing input is reported when the workchain starts, with the name of the input. If you skip the convergence, the GW mesh must be
commensurate with the Wannier90 mesh: otherwise the workchain stops with the exit code 401.
