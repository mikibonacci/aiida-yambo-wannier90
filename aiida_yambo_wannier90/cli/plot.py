#!/usr/bin/env python
"""Command to plot figures."""
import click
import matplotlib.pyplot as plt

from aiida import orm
from aiida.cmdline.params.types import NodeParamType
from aiida.cmdline.utils import decorators

from .root import cmd_root


@cmd_root.group("plot")
def cmd_plot():
    """Plot band structures of WorkChain."""


@cmd_plot.command("bands")
@click.argument("pw", type=NodeParamType())
@click.option(
    "--w90",
    type=NodeParamType(),
    help=(
        "If provided, use this BandsData instead of Wannier90 bands from W90_QP, "
        "accepts Wannier90BandsWorkChain/Wannier90BaseWorkChain/BandsData"
    ),
)
@click.argument("w90_qp", type=NodeParamType())
@decorators.with_dbenv()
def cmd_plot_bands(pw, w90, w90_qp):
    """Compare PW, Wannier90, and Wannier90 QP-corrected band structures.

    \b
    PW: PwBandsWorkChain/PwBaseWorkChain/BandsData
    W90_QP: YamboWannier90WorkChain/Wannier90BandsWorkChain/Wannier90BaseWorkChain/BandsData
    """
    # pylint: disable=import-outside-toplevel
    from aiida_wannier90_workflows.utils.workflows.plot import (
        get_band_dict,
        get_workchain_fermi_energy,
        get_workflow_output_band,
        plot_band,
        plot_bands_diff,
    )

    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    bands_pw = get_workflow_output_band(pw)

    if w90:
        bands_w90 = get_workflow_output_band(w90)

    if (
        hasattr(w90_qp, "process_class")
        and w90_qp.process_class == YamboWannier90WorkChain
    ):
        bands_w90_qp = w90_qp.outputs.band_structures.wannier90_qp
        if w90 is None:
            bands_w90 = w90_qp.outputs.band_structures.wannier90
    else:
        bands_w90_qp = get_workflow_output_band(w90_qp)

    _, ax = plt.subplots()

    fermi_energy = None
    if isinstance(pw, orm.WorkChainNode):
        fermi_energy = get_workchain_fermi_energy(pw)
    if fermi_energy is None:
        if isinstance(w90, orm.WorkChainNode):
            fermi_energy = get_workchain_fermi_energy(w90)

    print(f"{fermi_energy = }")

    plot_bands_diff(bands_pw, bands_w90, fermi_energy=fermi_energy, ax=ax)

    bands_w90_qp = get_band_dict(bands_w90_qp)
    bands_w90_qp["yaxis_label"] = "E (eV)"
    bands_w90_qp["legend_text"] = "W90_QP"
    bands_w90_qp["bands_color"] = "green"
    bands_w90_qp["bands_linestyle"] = "dashed"

    plot_band(bands_w90_qp, ref_zero=fermi_energy, ax=ax)

    ax.set_title(
        f"{pw.process_label}<{pw.pk}>,"
        f"{w90.process_label}<{w90.pk}>,"
        f"{w90_qp.process_label}<{w90_qp.pk}>"
    )

    plt.autoscale(axis="y")
    plt.show()
