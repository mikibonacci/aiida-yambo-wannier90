"""
Utils for builder.
"""
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import ProcessBuilder, ProcessBuilderNamespace

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
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

    if "yambo" in builder:
        if "scf" in builder["yambo"]["ywfl"]:
            set_parallelization(
                builder["yambo"]["ywfl"]["scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )
        if "nscf" in builder["yambo"]["ywfl"]:
            set_parallelization(
                builder["yambo"]["ywfl"]["nscf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

    if "yambo_qp" in builder:
        if "scf" in builder["yambo"]["ywfl"]:
            set_parallelization(
                builder["yambo_qp"]["scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )
        if "nscf" in builder["yambo"]["ywfl"]:
            set_parallelization(
                builder["yambo_qp"]["nscf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

    if "wannier90" in builder:
        set_parallelization(
            builder["wannier90"],
            parallelization=parallelization,
            process_class=Wannier90BandsWorkChain,
        )

    if "wannier90_qp" in builder:
        set_parallelization(
            builder["wannier90_qp"],
            parallelization=parallelization,
            process_class=Wannier90BaseWorkChain,
        )
