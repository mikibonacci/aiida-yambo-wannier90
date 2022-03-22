#!/usr/bin/env python
"""Run a ``PwBaseWorkChain`` for PW band structure.

Usage: ./example_02.py
"""
import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.types import FilteredWorkflowParamType
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    submit_and_add_group,
)
from aiida_wannier90_workflows.utils.workflows.builder.generator import (
    get_pwbands_builder,
)
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain


def submit(
    w90_wkchain: Wannier90BandsWorkChain, group: orm.Group = None, dry_run: bool = True
):
    """Submit a ``PwBaseWorkChain`` to calculate PW bands.

    Load a finished ``Wannier90BandsWorkChain``, and reuse the scf calculation.
    """
    builder = get_pwbands_builder(w90_wkchain)

    builder["pw"]["parallelization"] = orm.Dict(
        dict={
            "npool": 8,
        }
    )

    print_builder(builder)

    if not dry_run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.arguments.WORKFLOW(
    type=FilteredWorkflowParamType(
        process_classes=(
            "aiida.workflows:wannier90_workflows.bands",
            "aiida.workflows:wannier90_workflows.optimize",
        )
    ),
)
@click.option(
    "--run",
    "-r",
    is_flag=True,
    help="Submit workchain.",
)
@cmdline.params.options.GROUP(
    help="The group to add the submitted workchain.",
)
def cli(workflow, run, group):
    """Run a ``PwBaseWorkChain`` for PW band structure.

    Reuse the scf calculation from a finished Wannier90BandsWorkChain.
    """
    dry_run = not run

    # workflow = orm.load_node(139623)
    submit(workflow, group, dry_run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
