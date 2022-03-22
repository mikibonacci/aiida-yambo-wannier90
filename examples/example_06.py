#!/usr/bin/env python
"""Run a ``YamboWannier90WorkChain``.

Usage: ./example_06.py
"""
import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.kpoints import get_kpoints_from_bands
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    submit_and_add_group,
)


def submit(group: orm.Group = None, run: bool = False):
    """Submit a ``YamboWannier90WorkChain`` for the last step.

    Restart from a ``Gw2wannier90Calculation``.
    """
    # pylint: disable=import-outside-toplevel
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    codes = {
        "pw": "qe-git-pw@prnmarvelcompute5",
        "pw2wannier90": "qe-git-pw2wannier90@prnmarvelcompute5",
        "projwfc": "qe-git-projwfc@prnmarvelcompute5",
        "wannier90": "wannier90-git-wannier90@prnmarvelcompute5",
        "yambo": "yambo-5.0-yambo@prnmarvelcompute5",
        "p2y": "yambo-5.0-p2y@prnmarvelcompute5",
        "ypp": "yambo-5.0-ypp@prnmarvelcompute5",
        "gw2wannier90": "gw2wannier90@prnmarvelcompute5",
    }

    # Si2 from wannier90/example23
    w90_wkchain = orm.load_node(140073)  # Si
    structure = w90_wkchain.outputs.primitive_structure

    builder = YamboWannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
    )

    # Only run the 2nd half of the workchain: restart from an unsorted.eig
    del builder["yambo"]
    del builder["wannier90"]
    del builder["yambo_qp"]
    del builder["ypp"]

    # Reuse the inputs of a finished Gw2wannier90Calculation.
    gw2wan_calc = orm.load_node(140301)
    parent_folder = gw2wan_calc.inputs.parent_folder
    nnkp = gw2wan_calc.inputs.nnkp
    unsorted_eig = gw2wan_calc.inputs.unsorted_eig

    builder["gw2wannier90"]["parent_folder"] = parent_folder
    builder["gw2wannier90"]["nnkp"] = nnkp
    builder["gw2wannier90"]["unsorted_eig"] = unsorted_eig

    # Reuse wannier90 inputs
    w90calc = w90_wkchain.outputs.band_structure.creator
    # pylint: disable=protected-access
    w90calc_inputs = w90calc.inputs._construct_attribute_dict(incoming=True)
    # pylint: enable=protected-access
    del w90calc_inputs["structure"]
    del w90calc_inputs["kpoint_path"]
    del w90calc_inputs["remote_input_folder"]

    builder["wannier90_qp"]["wannier90"] = w90calc_inputs

    # Set `bands_kpoints` to skip seekpath step in YamboWannier90WorkChain
    builder.bands_kpoints = get_kpoints_from_bands(w90_wkchain.outputs.band_structure)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.GROUP(
    help="The group to add the submitted workchain.",
)
@RUN()
def cli(group, run):
    """Run a ``YamboWannier90WorkChain``."""
    submit(group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
