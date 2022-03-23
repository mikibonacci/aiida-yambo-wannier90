#!/usr/bin/env python
"""Launch a ``Gw2wannier90Calculation``.

Usage: ./example_05.py
"""
import pathlib

import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    submit_and_add_group,
)

from aiida_yambo_wannier90.calculations.gw2wannier90 import Gw2wannier90Calculation

INPUT_DIR = pathlib.Path(__file__).absolute().parent / "input_files" / "example_05"


def submit(group: orm.Group = None, run: bool = False):
    """Submit a ``Gw2wannier90Calculation``."""

    # Test Gw2wannier90Calculation with local inputs
    # code = orm.load_code("gw2wannier90@localhost")
    # parent_folder = orm.RemoteData(
    #     remote_path=str(INPUT_DIR / "unsorted"), computer=code.computer
    # )
    # nnkp = orm.SinglefileData(file=INPUT_DIR / "aiida.nnkp")
    # unsorted_eig = orm.SinglefileData(file=INPUT_DIR / "aiida.gw.unsorted.eig")

    code = orm.load_code("gw2wannier90@prnmarvelcompute5")
    w90_wkchain = orm.load_node(140073)  # Si
    parent_folder = w90_wkchain.outputs.wannier90.remote_folder
    nnkp = w90_wkchain.outputs.wannier90_pp.nnkp_file
    ypp_wkchain = orm.load_node(4048)
    unsorted_eig = ypp_wkchain.outputs.unsorted_eig_file

    builder = Gw2wannier90Calculation.get_builder()

    builder["code"] = code
    builder["parent_folder"] = parent_folder
    builder["nnkp"] = nnkp
    builder["unsorted_eig"] = unsorted_eig

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
    """Run a ``Gw2wannier90Calculation``."""
    submit(group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
