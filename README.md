# aiida-yambo-wannier90

GW-corrected, Wannier-interpolated band structures with [AiiDA](https://www.aiida.net): from a structure, the
`YamboWannier90WorkChain` runs the GW convergence and the quasiparticle calculation of
[Yambo](https://www.yambo-code.eu), the Wannierization of [Wannier90](https://wannier.org), and combines them. It builds on
`aiida-yambo`, `aiida-quantumespresso` and `aiida-wannier90-workflows`.

## Installation

Python >= 3.10, `aiida-core` 2.x and a recent `aiida-yambo` (>= 3.1):

```bash
pip install -e .
verdi plugin list aiida.workflows      # yambo_wannier90
```

Codes (`pw`, `pw2wannier90`, `wannier90`, `projwfc`, `yambo`, `p2y`, `ypp`, `gw2wannier90`) and the pseudopotentials are set up as
described in the documentation (`docs/installation.md`). The `gw2wannier90.py` script is installed with the
package: `python -c "from aiida_yambo_wannier90.scripts import GW2WANNIER90_SCRIPT; print(GW2WANNIER90_SCRIPT)"`.

## Usage

The protocol fills every stage; a few setters change what you need and keep the related inputs consistent:

```python
from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain
from aiida_yambo_wannier90.utils.setters import (
    print_summary, set_bands_kpath, set_qp_subsets, set_resources, set_wannier_mesh)

builder = YamboWannier90WorkChain.get_builder_from_protocol(codes, structure, protocol='moderate')
set_bands_kpath(builder, path=[('G', 'X'), ('X', 'U'), ('K', 'G')], point_coords=coords)   # or an explicit list of k-points
set_wannier_mesh(builder, [4, 4, 4])
set_qp_subsets(builder, qp_per_subset=50, parallel_runs=4, parallelism='auto')
set_resources(builder, 'gw', nodes=1, mpi=8, threads=2, time=4 * 3600)
print_summary(builder)
submit(builder)
```

The results are `band_structures.wannier90` (DFT) and `band_structures.wannier90_qp` (GW).

## Examples

Short linear scripts in [`examples/`](examples) (settings on top, builder, setters, `submit`), tested with test nodes:

1. `01_wannier90_dft_bands.py`: DFT bands interpolated with Wannier90.
2. `02_custom_kpath.py`: the band path: seekpath, distance, your path, labelled points, an explicit list.
3. `03_gw_wannier90_from_scratch.py`: the whole workchain, with the resources of every step.
4. `04_from_existing_gw.py`: start from a GW calculation, a QP database or a later stage.
5. `05_plot_results.py`: read and plot the bands.

## Tests

```bash
pip install -e .[testing]
pytest
```

The tests build the inputs and validate them against the process specifications; they do not run Quantum ESPRESSO,
Wannier90 or Yambo.
