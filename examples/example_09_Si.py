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
        "pw": "pw-6.8@hydralogin",
        "pw2wannier90": "pw2wannier90@hydralogin",
        "projwfc": "projwfc-6.8@hydralogin",
        "wannier90": "w90@hydralogin",
        "yambo": "yambo-RIMW@hydralogin",
        "p2y": "p2y-devel@hydralogin",
        "ypp": "ypp-RIMW@hydralogin",
        "gw2wannier90": "gw2wannier90@hydralogin",
    }

    # Si2 from wannier90/example23
    #w90_wkchain = orm.load_node(222)  # Si
    structure = orm.load_node(13496)  # Cu

    builder = YamboWannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        pseudo_family="PseudoDojo/0.4/LDA/SR/standard/upf",
    )

    # Increase ecutwfc
    params = builder.yambo.ywfl.scf.pw.parameters.get_dict()
    params["SYSTEM"]["ecutwfc"] = 100
    builder.yambo.ywfl.scf.pw.parameters = orm.Dict(dict=params)
    params = builder.yambo.ywfl.nscf.pw.parameters.get_dict()
    params["SYSTEM"]["ecutwfc"] = 100
    builder.yambo.ywfl.nscf.pw.parameters = orm.Dict(dict=params)

    parallelization = dict(
        max_wallclock_seconds=24 * 3600,
        # num_mpiprocs_per_machine=48,
        #npool=4,
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
        builder["yambo_qp"]["scf"],
        parallelization=parallelization,
        process_class=PwBaseWorkChain,
    )
    set_parallelization(
        builder["yambo_qp"]["nscf"],
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

    '''builder['yambo']['parameters_space']= orm.List(list=[{'conv_thr': 1,
                                 'conv_thr_units': '%',
                                 'convergence_algorithm': 'new_algorithm_1D',
                                 'delta': [2, 2, 2],
                                 'max': [32, 32, 32],
                                 'max_iterations': 4,
                                 'start': [8, 8, 8],
                                 'steps': 4,
                                 'stop': [16, 16, 16],
                                 'var': ['kpoint_mesh']},])'''
    
    builder['yambo']['workflow_settings']= orm.Dict(dict= {'bands_nscf_update': 'all-at-once',
                                 'skip_pre': True,
                                 'type': '1D_convergence',
                                 'what': ['gap_GG']},)


    builder['yambo']['ywfl']['yres']['yambo']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 4, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                                'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}

    builder['yambo']['ywfl']['scf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=16'}}
        
    builder['yambo']['ywfl']['nscf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 2, 'num_cores_per_mpiproc': 2, 'num_mpiprocs_per_machine':8}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=2'}}


    builder['yambo_qp']['additional_parsing'] = orm.List(list=['gap_GG'])

    builder['yambo_qp']['yres']['yambo']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                                'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}

    builder['yambo_qp']['scf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=16'}}
        
    builder['yambo_qp']['nscf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}

    
    builder['wannier90_qp']['wannier90']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 2, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=16'}}
    
    
    
    builder['wannier90']['nscf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines':4, 'num_cores_per_mpiproc': 1, 'num_mpiprocs_per_machine': 16}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=1'}}


    builder['wannier90']['scf']['pw']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=16'}}
    
    builder['wannier90']['pw2wannier90']['pw2wannier90']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 8, 'num_mpiprocs_per_machine':2}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=8'}}
    
    builder['wannier90']['wannier90']['wannier90']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 4, 'num_cores_per_mpiproc':2, 'num_mpiprocs_per_machine': 8}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': 'export OMP_NUM_THREADS=2'}}

    preprend_ypp_w = builder['ypp']['ypp']['metadata']['options']['prepend_text']
    builder['ypp']['ypp']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': preprend_ypp_w}}
    
    preprend_ypp_qp = builder['ypp_QP']['ypp']['metadata']['options']['prepend_text']
    builder['ypp_QP']['ypp']['metadata'] = {'options': {'stash': {}, 'resources': {'num_machines': 1, 'num_cores_per_mpiproc': 16, 'num_mpiprocs_per_machine': 1}, 'max_wallclock_seconds': 86400,
                                            'withmpi': True, 'queue_name': 's3par', 'prepend_text': preprend_ypp_qp}}

    #builder.pop('yambo')
    #builder['yambo_qp']['parent_folder'] = orm.load_node(13004).outputs.remote_folder
    #builder['yambo_qp']['QP_subset_dict'] = orm.Dict(dict={
    #                                    'qp_per_subset':50,
    #                                    'parallel_runs':4,
    #                                })

    '''builder['yambo']['ywfl']['yres']['yambo']['parameters'] = orm.Dict(dict={'arguments': ['dipoles', 'ppa', 'HF_and_locXC', 'gw0', 'NLCC', 'rim_cut'],
        'variables': {'Chimod': 'hartree',
        'DysSolver': 'n',
        'GTermKind': 'BG',
        'PAR_def_mode': 'workload',
        'X_and_IO_nCPU_LinAlg_INV': [1, ''],
        'RandQpts': [5000000, ''],
        'RandGvec': [100, 'RL'],
        'NGsBlkXp': [16, 'Ry'],
        'BndsRnXp': [[1, 400], ''],
        'GbndRnge': [[1, 400], ''],
        'QPkrange': [[[1, 1, 32, 32]], '']}})'''

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

