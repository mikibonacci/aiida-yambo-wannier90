# Choose the band path

`set_bands_kpath` chooses the path, one source at a time. More than one is an error, in the setter and when the workchain starts.

```python
--8<-- "examples/02_custom_kpath.py:kpath"
```

| Source | Meaning |
| --- | --- |
| nothing | seekpath finds the path and the primitive cell |
| `distance` | seekpath, with this reference distance (1/Å) between the points |
| `path` | segments of labels with the crystal coordinates of each label; wannier90 interpolates along them |
| `kpoints` | the same, as a labelled `KpointsData` of the high-symmetry points (input `kpoint_path`) |
| `explicit` | a labelled `KpointsData` list of points: the bands are computed exactly at these points (input `bands_kpoints`) |

With `path`, `kpoints` and `explicit` seekpath does not run, so the coordinates are those of **your** cell, not the primitive one. A
segment which starts where the previous one ended continues it; any other starts a new piece of the path (a jump). An explicit list is
used as it is in the DFT and in the GW wannier90 runs, so the two band structures are on the same points (for example to compare point
by point with the Quantum ESPRESSO bands).

The complete script, with all the modes (`KPATH_MODE`), is `examples/02_custom_kpath.py`.
