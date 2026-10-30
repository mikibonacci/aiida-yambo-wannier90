"""Read and plot the bands of a finished ``YamboWannier90WorkChain``: nothing is submitted.

``bands`` gives the arrays of a ``BandsData``; ``plot_bands`` draws the DFT interpolated bands and the GW-corrected ones (and,
if you give them, the Quantum ESPRESSO bands). The package also has a command::

    aiida-yambo-wannier90 plot bands <pw_pk> <yambo_wannier90_pk>

Set ``PK`` and run it with ``verdi run``, or import the functions.

Status: tested here on fabricated nodes.
"""
import numpy as np
from aiida import load_profile, orm

load_profile()

PK = None  # the pk of a finished YamboWannier90WorkChain
FILENAME = 'bands.png'


def bands(node):
    """The band structures of the workchain: ``{'wannier90': BandsData, 'wannier90_qp': BandsData}`` (what it has)."""
    result = {}
    for name in ('wannier90', 'wannier90_qp'):
        if name in node.outputs.band_structures:
            result[name] = node.outputs.band_structures[name]
    return result


def plot_bands(node, filename=FILENAME, reference=None):
    """Plot the bands of the workchain against the index of the k-point; `reference` is a `BandsData` (e.g. the PW bands)."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = {'wannier90': 'Wannier90 (DFT)', 'wannier90_qp': 'Wannier90 (GW)'}
    figure, axes = plt.subplots()
    for name, data in bands(node).items():
        values = np.asarray(data.get_bands())
        axes.plot(np.arange(len(values)), values, color='C0' if name == 'wannier90' else 'C3', lw=1)
        axes.plot([], [], color='C0' if name == 'wannier90' else 'C3', label=labels[name])
    if reference is not None:
        values = np.asarray(reference.get_bands())
        axes.plot(np.arange(len(values)), values, 'k.', ms=2)
        axes.plot([], [], 'k.', label='reference')
    axes.set_xlabel('k-point index')
    axes.set_ylabel('energy (eV)')
    axes.legend()
    figure.savefig(filename, dpi=120)
    return filename


if PK is not None:
    workchain = orm.load_node(PK)
    print({name: data.get_bands().shape for name, data in bands(workchain).items()})
    print('plot written:', plot_bands(workchain))
