# Examples

The scripts in `examples/` are short and linear (settings on top, protocol builder, setters, `submit`); edit the code labels and run
them with `verdi run`. They are run by the tests of the package with test nodes and a `submit` which checks the inputs against the
process specifications (nothing is run).

| Script | Shows |
| --- | --- |
| `01_wannier90_dft_bands.py` | DFT bands interpolated with Wannier90 (`Wannier90BandsWorkChain`), the first step alone |
| `02_custom_kpath.py` | the band path: seekpath, distance, your path, labelled points, an explicit list (`KPATH_MODE`) |
| `03_gw_wannier90_from_scratch.py` | the whole workchain: mesh, QP splitter, resources of every step, summary |
| `04_from_existing_gw.py` | start from a GW calculation, a QP database or a later stage (`STAGE`) |
| `05_plot_results.py` | read and plot the DFT and the GW interpolated bands |

The status of each script (run on a cluster, or only validated by the tests) is in its docstring; see also `to_test.md` for what is
still to be checked on a cluster. The GW convergence, the QP splitter and the resources are explained in the documentation of
`aiida-yambo`.
