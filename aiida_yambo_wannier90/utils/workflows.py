"""
Utils for workflows at runtime.
"""
from aiida import orm

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_yambo.utils.common_helpers import find_pw_parent
from aiida_yambo.workflows.yamboconvergence import YamboConvergence
from aiida_yambo.workflows.yambowf import YamboWorkflow


def _column(story, name):
    """The values of a column of the story, which is a dictionary of columns or a list of rows."""
    if isinstance(story, dict):
        column = story[name]
        return list(column.values()) if isinstance(column, dict) else list(column)
    return [row[name] for row in story]


def get_yambo_converged_workchain(workchain: YamboConvergence) -> YamboWorkflow:
    """Find converged ``YamboWorkflow`` from ``YamboConvergence``.

    :param workchain: A finished ``YamboConvergence``
    :type workchain: YamboConvergence
    :return: The converged ``YamboWorkflow``
    :rtype: YamboWorkflow
    """
    if workchain.process_class != YamboConvergence:
        raise ValueError(f"input workchain {workchain} is not a `YamboConvergence`")

    # `convergence_summary['uuid']` is the last converged calculation; if it is not there (the convergence of
    # the last parameter group did not finish), the last row of `collection['story']` flagged as converged.
    summary = workchain.outputs.convergence_summary.get_dict()
    uuid = summary.get("uuid")
    if uuid is None:
        story = workchain.outputs.collection.get_dict()["story"]
        converged = [
            (i, uid)
            for i, (uid, flag) in enumerate(zip(_column(story, "uuid"), _column(story, "converged")))
            if flag
        ]
        if not converged:
            raise ValueError(f"No converged `YamboWorkflow` in {workchain}")
        uuid = converged[-1][1]

    converged_wkchain = orm.load_node(uuid)

    return converged_wkchain


def get_yambo_nscf(workchain: YamboWorkflow) -> PwBaseWorkChain:
    """Find nscf ``PwBaseWorkChain`` in ``YamboWorkflow``.

    :param workchain: A finished ``YamboWorkflow``
    :type workchain: YamboWorkflow
    :return: The nscf ``PwBaseWorkChain``
    :rtype: PwBaseWorkChain
    """
    if workchain.process_class != YamboWorkflow:
        raise ValueError(f"input workchain {workchain} is not a `YamboWorkflow`")

    try:
        nscf_wkchain = (
            workchain.get_outgoing(
                link_label_filter="nscf",
            )
            .one()
            .node
        )
    except ValueError:
        # It can happen that the workflow has only as parent the nscf,
        # and did not run the calculation himself.
        # The find_pw_parent catches the pw, not the pw.base
        nscf_wkchain = find_pw_parent(workchain).caller

    return nscf_wkchain
