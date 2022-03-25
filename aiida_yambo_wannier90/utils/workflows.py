"""
Utils for workflows at runtime.
"""
from aiida import orm

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_yambo.workflows.yamboconvergence import YamboConvergence
from aiida_yambo.workflows.yambowf import YamboWorkflow


def get_yambo_converged_workchain(workchain: YamboConvergence) -> YamboWorkflow:
    """Find converged ``YamboWorkflow`` from ``YamboConvergence``.

    :param workchain: A finished ``YamboConvergence``
    :type workchain: YamboConvergence
    :return: The converged ``YamboWorkflow``
    :rtype: YamboWorkflow
    """
    if not isinstance(workchain, YamboConvergence):
        raise ValueError(f"input workchain {workchain} is not a `YamboConvergence`")

    yambo_history = workchain.outputs.history.get_dict()
    # e.g.
    #  'useful': {'0': False, '1': False, '2': False, '3': False, '4': True,
    #             '5': False, '6': False, '7': False, '8': False, '9': False}

    converged_idx = [k for k, v in yambo_history["useful"].items() if v]

    if len(converged_idx) < 1:
        raise ValueError(f"No converged `YamboWorkflow` in {workchain}")

    converged_idx = converged_idx[0]

    # The YamboWorkflow
    converged_wkchain = orm.load_node(yambo_history["uuid"][converged_idx])

    return converged_wkchain


def get_yambo_nscf(workchain: YamboWorkflow) -> PwBaseWorkChain:
    """Find nscf ``PwBaseWorkChain`` in ``YamboWorkflow``.

    :param workchain: A finished ``YamboWorkflow``
    :type workchain: YamboWorkflow
    :return: The nscf ``PwBaseWorkChain``
    :rtype: PwBaseWorkChain
    """
    if not isinstance(workchain, YamboWorkflow):
        raise ValueError(f"input workchain {workchain} is not a `YamboWorkflow`")

    nscf_wkchain = (
        workchain.get_outgoing(
            link_label_filter="nscf",
        )
        .one()
        .node
    )

    return nscf_wkchain
