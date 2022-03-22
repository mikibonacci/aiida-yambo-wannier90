#!/usr/bin/env python
"""Run a ``Wannier90BandsWorkChain``.

Usage: ./example_01.py
"""
import pathlib

import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints_from_mesh
from aiida_wannier90_workflows.utils.structure import read_structure
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    set_parallelization,
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain

INPUT_DIR = pathlib.Path(__file__).absolute().parent / "input_files" / "example_01"


def submit(group: orm.Group = None, dry_run: bool = True):
    """Submit a ``Wannier90BandsWorkChain``.

    Using parameters roughly the same as wannier90/example23.
    """
    codes = {
        "pw": "qe-git-pw@prnmarvelcompute5",
        "pw2wannier90": "qe-git-pw2wannier90@prnmarvelcompute5",
        "wannier90": "wannier90-git-wannier90@prnmarvelcompute5",
    }

    # Silicon
    structure = read_structure(INPUT_DIR / "Si2.cif")
    # structure = orm.load_node(139524)

    # Run a Wannier90BandsWorkChain with the same params as example23
    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        projection_type=WannierProjectionType.ANALYTIC,
        # Use NCPP in Yambo
        pseudo_family="PseudoDojo/0.4/PBE/SR/standard/upf",
    )

    # Use 4x4x4 kmesh
    kpoints = get_explicit_kpoints_from_mesh(structure, [4, 4, 4])
    builder.wannier90.wannier90.kpoints = kpoints
    builder.nscf.kpoints = kpoints

    # Change number of bands as example23
    parameters = builder.nscf.pw.parameters.get_dict()
    parameters["SYSTEM"]["nbnd"] = 14
    builder.nscf.pw.parameters = orm.Dict(dict=parameters)
    # Change wannier90 num_bands
    parameters = builder.wannier90.wannier90.parameters.get_dict()
    parameters["num_bands"] = 14
    parameters["mp_grid"] = [4, 4, 4]
    builder.wannier90.wannier90.parameters = orm.Dict(dict=parameters)

    # Set parallelization
    parallelization = {
        "npool": 1,
        "num_mpiprocs_per_machine": 8,
        # 'npool': 8,
        # 'num_mpiprocs_per_machine': 48,
    }
    builder = set_parallelization(builder, parallelization)

    print_builder(builder)

    if not dry_run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@click.option(
    "--run",
    "-r",
    is_flag=True,
    help="Submit workchain.",
)
@cmdline.params.options.GROUP(
    help="The group to add the submitted workchain.",
)
def cli(run, group):
    """Run a ``Wannier90BandsWorkChain``."""
    dry_run = not run
    submit(group, dry_run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
