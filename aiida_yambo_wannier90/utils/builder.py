"""
Utils for builder.
"""
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import ProcessBuilder, ProcessBuilderNamespace

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain

from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain


def set_parallelization(  # pylint: disable=too-many-locals,too-many-statements
    builder: ty.Union[ProcessBuilder, ProcessBuilderNamespace, AttributeDict],
    parallelization: dict = None,
    process_class: orm.WorkChainNode = YamboWannier90WorkChain,
) -> None:
    """Set MPI, npool for ``YamboWannier90WorkChain``.

    :param builder: a builder or its subport, or a ``AttributeDict`` which is the inputs for the builder.
    :type builder: ProcessBuilderNamespace
    :param parallelization: _description_, defaults to None
    :type parallelization: dict, optional
    :param process_class: _description_, defaults to YamboWannier90WorkChain
    :type process_class: orm.WorkChainNode, optional
    """
    from aiida_wannier90_workflows.utils.workflows.builder import (
        set_parallelization as set_para,
    )

    if process_class != YamboWannier90WorkChain:
        set_para(builder, parallelization, process_class)
        return

    # the DFT steps of the GW calculations: `yambo` (convergence, namespace `ywfl`) and `yambo_qp` (a `YamboWorkflow`)
    gw_namespaces = []
    if "yambo" in builder and "ywfl" in builder["yambo"]:
        gw_namespaces.append(builder["yambo"]["ywfl"])
    if "yambo_qp" in builder:
        gw_namespaces.append(builder["yambo_qp"])
    for namespace in gw_namespaces:
        for step in ("scf", "nscf"):
            if step in namespace:
                set_parallelization(
                    namespace[step],
                    parallelization=parallelization,
                    process_class=PwBaseWorkChain,
                )

    if "wannier90" in builder:
        set_parallelization(
            builder["wannier90"],
            parallelization=parallelization,
            process_class=Wannier90OptimizeWorkChain,
        )

    if "wannier90_qp" in builder:
        set_parallelization(
            builder["wannier90_qp"],
            parallelization=parallelization,
            process_class=Wannier90BaseWorkChain,
        )
