# aiida-yambo-wannier90

GW-corrected, Wannier-interpolated band structures with [AiiDA](https://www.aiida.net). From a structure, the
`YamboWannier90WorkChain` runs the GW convergence and the quasiparticle calculation of
[Yambo](https://www.yambo-code.eu), the Wannierization of [Wannier90](https://wannier.org), and combines them. It builds on
`aiida-yambo`, `aiida-quantumespresso` and `aiida-wannier90-workflows`.

<div class="grid cards" markdown>

-   **Install and set up the codes**

    Python 3.10+, the eight codes, the pseudopotentials.

    [Installation](installation.md)

-   **How it works**

    The stages, the two k-meshes, the two Wannier90 runs.

    [The workchain](concepts/workflow.md)

-   **Change the inputs**

    The builder and the setters: band path, meshes, bands, QP splitter, resources, starting from the middle.

    [How-to](how-to/index.md)

-   **Examples**

    Five short, tested scripts.

    [Examples](examples.md)

</div>

```python
from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain
from aiida_yambo_wannier90.utils.setters import print_summary, set_bands_kpath, set_resources, set_wannier_mesh

builder = YamboWannier90WorkChain.get_builder_from_protocol(codes, structure, protocol='moderate')
set_bands_kpath(builder, path=[('G', 'X'), ('X', 'U'), ('K', 'G')], point_coords=coords)  # or an explicit list of points
set_wannier_mesh(builder, [4, 4, 4])
set_resources(builder, 'gw', nodes=1, mpi=8, threads=2, time=4 * 3600)
print_summary(builder)
```

The results are `band_structures.wannier90` (DFT) and `band_structures.wannier90_qp` (GW).
