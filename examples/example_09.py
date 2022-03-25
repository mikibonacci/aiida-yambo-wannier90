#!/usr/bin/env python
"""Run a full ``YamboWannier90WorkChain``.

Usage: ./example_09.py
"""
import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    submit_and_add_group,
)


def submit(group: orm.Group = None, run: bool = False):
    """Submit a ``YamboWannier90WorkChain`` from scratch.

    Run all the steps.
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
