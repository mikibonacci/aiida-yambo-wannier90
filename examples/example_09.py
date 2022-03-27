#!/usr/bin/env python
"""Run a full ``YamboWannier90WorkChain``.

Usage: ./example_09.py

To compare bands between QE, W90, W90 with QP, run in terminal
```
aiida-yambo-wannier90 plot bands PW_PK GWW90_PK
```
Where `PW_PK` is the PK of a `PwBandsWorkChain/PwBaseWorkChain` for PW bands calculation,
`GWW90_PK` is the PK of a `YamboWannier90WorkChain`.
"""
import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    set_parallelization,
    submit_and_add_group,
)


def submit(group: orm.Group = None, run: bool = False):
    """Submit a ``YamboWannier90WorkChain`` from scratch.

    Run all the steps.
    """
    # pylint: disable=import-outside-toplevel
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
    from aiida_wannier90_workflows.workflows.base.wannier90 import (
        Wannier90BaseWorkChain,
    )

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

    # Increase ecutwfc
    params = builder.yambo.ywfl.scf.pw.parameters.get_dict()
    params["SYSTEM"]["ecutwfc"] = 80
    builder.yambo.ywfl.scf.pw.parameters = orm.Dict(dict=params)
    params = builder.yambo.ywfl.nscf.pw.parameters.get_dict()
    params["SYSTEM"]["ecutwfc"] = 80
    builder.yambo.ywfl.nscf.pw.parameters = orm.Dict(dict=params)

    parallelization = dict(
        max_wallclock_seconds=24 * 3600,
        # num_mpiprocs_per_machine=48,
        npool=8,
        num_machines=1,
    )
    set_parallelization(
        builder["yambo"]["ywfl"]["scf"],
        parallelization=parallelization,
        process_class=PwBaseWorkChain,
    )
    set_parallelization(
        builder["yambo"]["ywfl"]["nscf"],
        parallelization=parallelization,
        process_class=PwBaseWorkChain,
    )
    set_parallelization(
        builder["wannier90"],
        parallelization=parallelization,
        process_class=Wannier90BandsWorkChain,
    )
    set_parallelization(
        builder["wannier90_qp"],
        parallelization=parallelization,
        process_class=Wannier90BaseWorkChain,
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
