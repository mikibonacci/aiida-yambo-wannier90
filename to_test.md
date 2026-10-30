# To test on a cluster

The test suite (`pytest`) builds the inputs and validates them against the process specifications, with fake codes: nothing was run
with Quantum ESPRESSO, Wannier90, Yambo or `gw2wannier90.py` after the modernisation. This is the list of what to check, in order, with
what to look at and what I am least sure about. Use silicon, the `fast` protocol and a small mesh first.

How to look: `verdi process status <pk>`, `verdi process report <pk>` (the workchain and each child), `verdi process show <pk>`
(inputs and outputs). The scripts are in `examples/`; edit the code labels at the top.

## 0. Setup

- [ ] `pip install -e .` in an environment with `aiida-yambo` >= 3.1 (the branch with `qp`/`bse` namespaces), `aiida-wannier90-workflows` 2.7,
      `aiida-quantumespresso`; `verdi plugin list aiida.workflows` lists `yambo_wannier90`.
- [ ] The eight codes exist (`docs/installation.md`). `gw2wannier90`: its `remote_abs_path` is the output of
      `python -c "from aiida_yambo_wannier90.scripts import GW2WANNIER90_SCRIPT; print(GW2WANNIER90_SCRIPT)"`; on a remote cluster the script is
      copied there and is executable.
- [ ] `pytest` passes in your environment (`pip install -e .[testing]`).

## 1. `examples/01_wannier90_dft_bands.py` (cheap, no Yambo)

- [ ] Wannier90 bands are produced. This only uses `aiida-wannier90-workflows` and tells you the Wannier functions are good before GW.
- [ ] Keep the `remote_folder` of the wannier90 calculation for the next tests.

## 2. `examples/02_custom_kpath.py` (the new k-path code)

The path only matters from the Wannier stages on, but the stages before cannot be skipped from scratch, so check it on the runs of section 3
(or of section 4, which is cheaper once you have a GW calculation) with a different `KPATH_MODE` each time:

- [ ] `seekpath` (default): the structure of the Wannier stages is the primitive cell (`primitive_structure` output).
- [ ] `path`: no seekpath step; in the wannier90 calculations `verdi calcjob inputcat` shows a `kpoint_path` block with your segments
      (including a jump, e.g. `X-U` then `K-G`).
- [ ] `kpoints`: the same result as `path` with the same points.
- [ ] `explicit`: the `.win` of **both** wannier90 runs (DFT and GW) has an `explicit_kpath` block with your points and labels, and the two
      `interpolated_bands` outputs have the same number of k-points. This is the case that did not work before.
- [ ] `distance`: the number of k-points on the path changes with `DISTANCE`.
- [ ] The workchain refuses two sources at once (`set_bands_kpath` raises; also when you set two inputs by hand).
- [ ] The coordinates you give are used as crystal coordinates of YOUR cell: check one point (e.g. X) is where you expect in the band plot.

## 3. `examples/03_gw_wannier90_from_scratch.py` (the whole workchain)

Check every stage in order in `verdi process status`:

- [ ] **seekpath** and **yambo (convergence)**: finishes; `convergence_summary` has a `uuid`; the converged mesh is reported.
- [ ] **kmesh**: the report says `Found commensurate meshes GW = ..., W90 = ...`. If you chose a Wannier mesh incommensurate with the GW one, the
      GW mesh is adapted (or the run stops with 401 when you skipped the convergence).
- [ ] **yambo_commensurate** (only if the mesh changed): a `YamboWorkflow` on the commensurate mesh.
- [ ] **yambo_qp** — the part I am least sure about, check carefully:
  - [ ] it starts from the converged calculation (`parent_folder`) and does not repeat the SCF/NSCF unless the mesh changed;
  - [ ] its `qp.yambo.parameters` are the converged ones, and `QP_subsets_dict` is `{'qp_per_subset': .., 'parallel_runs': .., 'explicit': [...]}`
        with the (k, band) ranges of the coarse mesh (`kmapper` output in the provenance);
  - [ ] the first plain `qp` run, the splitted runs (`yambo_QP_splitted_*`), `merged_QP` exist; with `parallelism='auto'` the splitted
        runs have `*_CPU`/`*_ROLEs` variables;
  - [ ] the settings `COPY_SAVE`/`COPY_DBS` of the QP calculation did not break the restart.
- [ ] **wannier90_pp**: `nnkp` file output.
- [ ] **ypp**: `unsorted.eig` is produced. **Suspicious line**: the `parent_folder` of `ypp` is
      `wkchain_yambo_qp.outputs.remote_folder.creator.outputs.remote_folder`, which is just the remote folder of the last Yambo calculation of the
      QP workflow. With the QP splitter this is a splitted run, not obviously the folder with the full `SAVE`: if `ypp` fails or gives the wrong
      energies, look here first (`aiida_yambo_wannier90/workflows/__init__.py`, `prepare_ypp_inputs`).
- [ ] **wannier90** (DFT): `band_structures.wannier90`.
- [ ] **gw2wannier90**: runs the script; `output_parameters` and `sort_index`.
- [ ] **wannier90_qp**: uses the same `num_bands`, `num_wann`, `exclude_bands` and mesh as the DFT run (see the `.win` files), and
      `band_structures.wannier90_qp` exists.
- [ ] **Physics**: the GW bands differ from the DFT ones (a gap opening for silicon), and the gap at the k-points of the coarse mesh agrees with
      the QP gap computed by Yambo (`output_ywfl_parameters` of `yambo_qp`).
- [ ] `kmapper`: `start_band` is `max(exclude_bands) + 1` and `end_band` follows `num_bands`: check the bands in the QP ranges are the Wannier ones
      (matters when `exclude_semicore` / `exclude_bands` is used).

## 4. `examples/04_from_existing_gw.py` (starting from the middle)

Use the results of section 3 (the pks). Each reads inputs of earlier stages and several steps of the workchain walk the provenance
(`parent_folder.creator.caller...`), which is fragile:

- [ ] `STAGE = 'yambo_qp'` with `PARENT_PK` = the remote folder of the converged GW `YamboWorkflow`: `setup` finds the GW mesh from the parent
      (look for an `AttributeError`/`NoneType` in the report).
- [ ] `STAGE = 'yambo_qp'` with `GW_MESH`: runs `yambo_commensurate` first.
- [ ] `STAGE = 'ypp'` with the QP calculation remote folder and the merged `ndb.QP`.
- [ ] `STAGE = 'wannier90_qp'` with the remote folder of `gw2wannier90`.
- [ ] A missing input is reported with a clear message when the workchain starts.

## 5. The setters on a real run

- [ ] `print_summary(builder)` shows what you expect (stages, mesh, bands, projections, resources).
- [ ] `set_resources(builder, 'gw', nodes=, mpi=, threads=, time=)`: `verdi calcjob inputcat <pk>` of a GW calculation shows the resources and
      `export OMP_NUM_THREADS=<threads>`; the same in the convergence and in the QP calculation. `npool` appears in the `pw.x` command line.
- [ ] `set_wannier_mesh`, `set_wannier_bands`, `set_projections`: change them, check the DFT and the GW wannier90 inputs agree (`mp_grid`,
      `num_wann`, `num_bands`, `exclude_bands`, projections).
- [ ] `set_qp_subsets(..., parallelism='auto', resources=..., max_wallclock_seconds=...)` reach the splitted runs (these are the new
      QP-splitter options of `aiida-yambo`).
- [ ] `overrides={'yambo': {'resource_rules': ...}}` is accepted and applied during the convergence (new in `aiida-yambo`: untested together).

## 6. Variations of the protocol (each is a new run)

- [ ] `protocol='moderate'` and `'precise'`: the GW and the Wannier90 stages both change (more bands, tighter parameters).
- [ ] `electronic_type=INSULATOR` for silicon: no disentanglement, `num_wann = num_bands`; compare with METAL (the default, which disentangles).
- [ ] `wannier_projection_type`: `ANALYTIC` (then `set_projections`), `SCDM`: both wannier90 runs get the same types.
- [ ] A metal (Cu) and an insulator with several species (hBN, GaAs): the `QP_subsets_dict` and `kmapper` with more bands, `exclude_semicore`.
- [ ] `kpoints_force_gw=True` (`set_gw_mesh(..., force=True)`).

## 7. After a run

- [ ] `examples/05_plot_results.py` (set `PK`): the plot of DFT and GW bands.
- [ ] `aiida-yambo-wannier90 plot bands <pw_pk> <workchain_pk>`: the CLI command was not changed or tested; it reads
      `band_structures.wannier90_qp`.
- [ ] `clean_workdir=True` at the end cleans the remote folders (including the splitted runs).

## What is known and not done

- The notebooks in `tutorials-aiida-yambo/yambo_wannier90` still use the old API (`yres`, `QP_subset_dict`).
- `YamboWannier90WorkChain` still walks the provenance in a few places (`setup`, `prepare_yambo_qp_inputs`, `prepare_ypp_inputs`) assuming a
  `YamboWorkflow` parent: with data from other sources it can fail. The error is then an `AttributeError`/`NoneType`, not a clean message.
- The module `workflows/__init__.py` is still long: only the validators were moved out.
- The default `electronic_type` is METAL (disentanglement on), unchanged from before.
