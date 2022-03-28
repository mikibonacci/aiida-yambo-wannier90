#!/usr/bin/env python
"""Command to plot figures."""
import click
import matplotlib.pyplot as plt

from aiida import orm

# from aiida.cmdline.params.types import NodeParamType
from aiida.cmdline.utils import decorators

from aiida_wannier90_workflows.cli.params import FilteredWorkflowParamType

from .root import cmd_root


@cmd_root.group("plot")
def cmd_plot():
    """Plot band structures of WorkChain."""


@cmd_plot.command("bands")
@click.argument(
    "pw",
    type=FilteredWorkflowParamType(
        process_classes=(
            "aiida.workflows:quantumespresso.pw.base",
            "aiida.workflows:quantumespresso.pw.bands",
        )
    ),
)
@click.option(
    "--w90",
    type=FilteredWorkflowParamType(
        process_classes=(
            "aiida.workflows:wannier90_workflows.base.wannier90",
            "aiida.workflows:wannier90_workflows.bands",
        )
    ),
    help=(
        "If provided, use this BandsData instead of Wannier90 bands from W90_QP, "
        "accepts Wannier90BandsWorkChain/Wannier90BaseWorkChain/BandsData"
    ),
)
@click.argument(
    "w90_qp",  # pylint: disable=too-many-locals
    type=FilteredWorkflowParamType(
        process_classes=(
            "aiida.workflows:yambo_wannier90",
            "aiida.workflows:wannier90_workflows.base.wannier90",
        )
    ),
)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    default=False,
    help="Save figure instead of showing matplotlib window",
)
@click.option(
    "-f",
    "--filename",
    type=str,
    default=None,
    help="Filename of the saved figure, default format is PNG.",
)
@decorators.with_dbenv()
def cmd_plot_bands(pw, w90, w90_qp, save, filename):
    """Compare PW, Wannier90, and Wannier90 QP-corrected band structures.

    \b
    PW: PwBandsWorkChain/PwBaseWorkChain/BandsData
    W90_QP: YamboWannier90WorkChain/Wannier90BandsWorkChain/Wannier90BaseWorkChain/BandsData
    """
    # pylint: disable=import-outside-toplevel,too-many-locals,too-many-branches
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
        if isinstance(w90_qp, orm.WorkChainNode):
            if w90_qp.process_class == YamboWannier90WorkChain:
                calc = w90_qp.get_outgoing(link_label_filter="wannier90_qp").one().node
                fermi_energy = get_workchain_fermi_energy(calc)
            else:  # e.g. Wannier90BaseWorkChain
                fermi_energy = get_workchain_fermi_energy(w90_qp)

    print(f"{fermi_energy = }")

    plot_bands_diff(bands_pw, bands_w90, fermi_energy=fermi_energy, ax=ax)

    bands_w90_qp = get_band_dict(bands_w90_qp)
    bands_w90_qp["yaxis_label"] = "E (eV)"
    bands_w90_qp["legend_text"] = "W90_QP"
    bands_w90_qp["bands_color"] = "green"
    bands_w90_qp["bands_linestyle"] = "dashed"

    plot_band(bands_w90_qp, ref_zero=fermi_energy, ax=ax)

    title = ""
    if w90 is None:
        title = f"{pw.process_label}<{pw.pk}>,{w90_qp.process_label}<{w90_qp.pk}>"
    else:
        title = (
            f"{pw.process_label}<{pw.pk}>,"
            f"{w90.process_label}<{w90.pk}>,"
            f"{w90_qp.process_label}<{w90_qp.pk}>"
        )
    ax.set_title(title)

    plt.autoscale(axis="y")

    if save:
        if filename is None:
            if w90 is None:
                filename = f"bandsdiff-PW_{pw.pk}-GWW90_{w90_qp.pk}.png"
            else:
                filename = f"bandsdiff-PW_{pw.pk}-W90_{w90.pk}-GWW90_{w90_qp.pk}.png"
        plt.savefig(filename)
        print(f"Saved to {filename}")
    else:
        plt.show()
