#!/usr/bin/env python
"""Run a ``YamboWannier90WorkChain``.

Usage: ./example_08.py

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
from aiida_wannier90_workflows.utils.kpoints import get_kpoints_from_bands
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    set_parallelization,
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain


def submit(group: orm.Group = None, run: bool = False):
    """Submit a ``YamboWannier90WorkChain`` starting from ``wannier90`` step.

    Skipping the 1st ``YamboConvergence`` step.
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

    # Set `bands_kpoints` to skip seekpath step
    builder.bands_kpoints = get_kpoints_from_bands(w90_wkchain.outputs.band_structure)

    # Skip YamboConvergence
    del builder["yambo"]

    # Reuse a YamboWorkflow as the parent_folder for yambo_qp
    yambo_wkchain = orm.load_node(140260)
    builder.yambo_qp.parent_folder = yambo_wkchain.outputs.remote_folder
    # Reuse parameters
    params = yambo_wkchain.inputs.yres.yambo.parameters.get_dict()
    params["variables"]["X_and_IO_nCPU_LinAlg_INV"] = [1, ""]
    builder.yambo_qp.yres.yambo.parameters = orm.Dict(dict=params)

    # Force W90 use the same GW kmesh
    # (some times W90 mesh is higher than GW converged)
    # 1. either set kpoints explicitly
    # from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain
    # from aiida_wannier90_workflows.utils.workflows.builder import set_kpoints
    # kpoints_gw = yambo_wkchain.inputs.nscf.kpoints
    # # wannier90 step
    # set_kpoints(builder["wannier90"], kpoints_gw, process_class=Wannier90BandsWorkChain)
    # # wannier90_qp step
    # set_kpoints(
    #     builder["wannier90_qp"], kpoints_gw, process_class=Wannier90BaseWorkChain
    # )
    # 2. or using builder.kpoints_force_gw
    builder.kpoints_force_gw = True

    parallelization = dict(
        max_wallclock_seconds=5 * 3600,
        num_mpiprocs_per_machine=48,
        npool=8,
        num_machines=1,
    )
    set_parallelization(
        builder["wannier90"],
        parallelization=parallelization,
        process_class=Wannier90BandsWorkChain,
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
