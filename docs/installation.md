# Installation

It needs Python 3.10 or newer, `aiida-core` 2.x and a recent `aiida-yambo` (3.1 or newer: the version with the `qp` and `bse`
namespaces).

```bash
git clone https://github.com/aiidaplugins/aiida-yambo-wannier90
pip install -e ./aiida-yambo-wannier90
verdi plugin list aiida.workflows      # should list yambo_wannier90
```

A configured AiiDA profile and a running daemon (`verdi status`) are needed to run anything.

## Codes

Eight codes, by label (`verdi code list`):

| Key | Program | Input plugin |
| --- | --- | --- |
| `pw` | Quantum ESPRESSO `pw.x` | `quantumespresso.pw` |
| `pw2wannier90` | `pw2wannier90.x` | `quantumespresso.pw2wannier90` |
| `wannier90` | `wannier90.x` | `wannier90.wannier90` |
| `projwfc` | `projwfc.x` (atomic projectors) | `quantumespresso.projwfc` |
| `yambo` | `yambo` | `yambo.yambo` |
| `p2y` | `p2y` | `yambo.p2y` |
| `ypp` | `ypp` | `yambo.ypp` |
| `gw2wannier90` | the script `gw2wannier90.py` | `yambo_wannier90.gw2wannier90` |

`gw2wannier90.py` is installed with the package: it is a Python script which the calculation runs on the computer. Print its
path and use it as the `remote_abs_path` of the code. `aiida_yambo_wannier90/scripts/gw2wannier90@localhost.yaml` is a template for
`verdi code setup --config`; on a remote cluster copy the script there first.

```bash
python -c "from aiida_yambo_wannier90.scripts import GW2WANNIER90_SCRIPT; print(GW2WANNIER90_SCRIPT)"
```

## Pseudopotentials

Yambo needs norm-conserving pseudopotentials: `aiida-pseudo install pseudo-dojo`. The family
`PseudoDojo/0.4/PBE/SR/standard/upf` is the default.

## A first run

The scripts in `examples/` are short and linear: settings on top, the protocol builder, a few setters, `submit`. Edit the code
labels and run one:

```bash
verdi daemon start
verdi run examples/03_gw_wannier90_from_scratch.py
verdi process status <pk>
```

See [Examples](examples.md) and [Set the inputs](how-to/inputs.md).
