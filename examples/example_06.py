#!/usr/bin/env python
"""Run a ``YamboWannier90WorkChain``, restart from ``unsorted.eig``.

Usage: ./example_06.py
"""
import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.kpoints import (
    get_explicit_kpoints_from_mesh,
    get_kpoints_from_bands,
)
from aiida_wannier90_workflows.utils.workflows.builder import (
    get_metadata,
    print_builder,
    set_kpoints,
    set_num_bands,
    set_parallelization,
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain


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
        # daint
        # "pw": "qe-6.8-pw@daint",
        # "pw2wannier90": "qe-6.8-pw2wannier90@daint",
        # "wannier90": "wannier90-3.1-wannier90@daint",
        # "yambo": "yambo-5.0-yambo@daint",
        # "p2y": "yambo-5.0-p2y@daint",
        # "ypp": "yambo-5.0-ypp@daint",
        # "gw2wannier90": "gw2wannier90@daint",
        # lumi
        # "pw": "qe-6.8-pw@lumi-small",
        # "pw2wannier90": "qe-6.8-pw2wannier90@lumi-small",
        # "wannier90": "wannier90-git-wannier90@lumi-small",
        # "yambo": "yambo-5.1-yambo@lumi-small",
        # "p2y": "yambo-5.1-p2y@lumi-small",
        # "ypp": "yambo-5.1-ypp@lumi-small",
        # "gw2wannier90": "gw2wannier90@lumi-small",
    }

    # Si2 from wannier90/example23
    w90_wkchain = orm.load_node(140073)  # Si
    # w90_wkchain = orm.load_node(9623)  # PdCoO2
    structure = w90_wkchain.outputs.primitive_structure

    builder = YamboWannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
    )

    # Only run the 2nd half of the workchain: restart from an unsorted.eig
    del builder["yambo"]
    del builder["yambo_qp"]
    del builder["ypp"]

    # Silicon, reuse the inputs of a finished Gw2wannier90Calculation.
    gw2wan_calc = orm.load_node(140301)
    unsorted_eig = gw2wan_calc.inputs.unsorted_eig

    # PdCoO2, restart from ypp output unsorted.eig
    # ypp_wkchain = orm.load_node(9375)
    # unsorted_eig = ypp_wkchain.outputs.unsorted_eig_file

    builder["gw2wannier90"]["unsorted_eig"] = unsorted_eig

    # Set `bands_kpoints` to skip seekpath step in YamboWannier90WorkChain
    builder.bands_kpoints = get_kpoints_from_bands(w90_wkchain.outputs.band_structure)

    # Use 4x4x4 kmesh from unsorted.eig
    kpoints = get_explicit_kpoints_from_mesh(structure, [4, 4, 4])
    set_kpoints(builder["wannier90"], kpoints, process_class=Wannier90BandsWorkChain)
    set_kpoints(builder["wannier90_qp"], kpoints, process_class=Wannier90BaseWorkChain)

    # Change number of bands as example23
    num_bands = 14
    set_num_bands(
        builder["wannier90"], num_bands, process_class=Wannier90BandsWorkChain
    )
    set_num_bands(
        builder["wannier90_qp"], num_bands, process_class=Wannier90BaseWorkChain
    )

    # Set parallelization
    parallelization = {
        # prn
        "max_wallclock_seconds": 24 * 3600,
        "num_machines": 1,
        "npool": 8 * 1,
        "num_mpiprocs_per_machine": 48,
        # daint
        #'max_wallclock_seconds': 1800,
        #'num_machines': 10,
        #'npool': 3*10,
        #'num_mpiprocs_per_machine': 12,
        #'queue_name': 'debug',
        #'account': 'mr0',
        # lumi
        # 'max_wallclock_seconds': 24 * 3600,
        # 'num_machines': 1,
        # 'npool': 16*1,
        # 'num_mpiprocs_per_machine': 128,
    }
    set_parallelization(
        builder["wannier90"], parallelization, process_class=Wannier90BandsWorkChain
    )
    set_parallelization(
        builder["wannier90_qp"], parallelization, process_class=Wannier90BaseWorkChain
    )
    builder["gw2wannier90"]["metadata"] = get_metadata(**parallelization)

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
