#!/usr/bin/env python
"""Base class for Yambo+Wannier90 workflow."""
import pathlib
import typing as ty

import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ExitCode, ProcessBuilder, ToContext, WorkChain, if_

from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import (
    seekpath_structure_analysis,
)
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_wannier90_workflows.utils.kpoints import (
    create_kpoints_from_mesh,
    get_explicit_kpoints_from_mesh,
    get_mesh_from_kpoints,
)
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90BaseWorkChain,
    Wannier90OptimizeWorkChain,
)

from aiida_yambo.workflows.yamboconvergence import YamboConvergence
from aiida_yambo.workflows.yambowf import YamboWorkflow
from aiida_yambo.workflows.ypprestart import YppRestart

from aiida_yambo_wannier90.calculations.functions.kmesh import (
    find_commensurate_meshes,
    get_output_explicit_kpoints,
    kmapper,
)
from aiida_yambo_wannier90.calculations.gw2wannier90 import Gw2wannier90Calculation

__all__ = ["validate_inputs", "YamboWannier90WorkChain"]


def validate_inputs(
    inputs: AttributeDict, ctx=None  # pylint: disable=unused-argument
) -> None:
    """Validate the inputs of the entire input namespace."""

    required_namespaces = ["yambo", "yambo_qp"]
    if all(_ not in inputs for _ in required_namespaces):
        raise ValueError(
            f"At least one of the {required_namespaces} should be provided."
        )

    required_namespaces = ["wannier90", "wannier90_qp"]
    if all(_ not in inputs for _ in required_namespaces):
        raise ValueError(
            f"At least one of the {required_namespaces} should be provided."
        )

    order = ["yambo", "wannier90", "yambo_qp", "ypp", "gw2wannier90", "wannier90_qp"]
    presented = [_ in inputs for _ in order]
    first_input = presented.index(True)
    if first_input != (len(presented) - 1):
        not_presented = [not _ for _ in presented]
        if any(not_presented[first_input + 1 :]):
            first_no_input = first_input + 1
            first_no_input += not_presented[first_input + 1 :].index(True)
            # TODO Two special cases,
            # yambo_qp can be auto-generated from yambo
            # wannier90_qp can be auto-generated from wannier90
            raise ValueError(
                f"Workchain must be run in order, {order[first_input]} is provided "
                f"but {order[first_no_input]} is absent."
            )

    if "wannier" not in inputs:
        if "nnkp" not in inputs.gw2wannier90:
            raise ValueError(
                "`inputs.wannier` is absent and no `nnkp` in `inputs.gw2wannier90`"
            )

    if "ypp" not in inputs:
        if "unsorted_eig" not in inputs.gw2wannier90:
            raise ValueError(
                "`inputs.ypp` is absent and no `unsorted_eig` in `inputs.gw2wannier90`"
            )


class YamboWannier90WorkChain(ProtocolMixin, WorkChain):
    """Workchain to obtain GW-corrected maximally localised Wannier functions (MLWF)."""

    @classmethod
    def define(cls, spec):
        """Define the process spec."""

        super().define(spec)

        spec.input(
            "structure", valid_type=orm.StructureData, help="The input structure."
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            serializer=orm.to_aiida_type,
            default=lambda: orm.Bool(False),
            help=(
                "If True, work directories of all called calculation will be cleaned "
                "at the end of execution."
            ),
        )
        spec.input(
            "bands_kpoints",
            valid_type=orm.KpointsData,
            required=False,
            help=(
                "Explicit kpoints to use for the band structure. "
                "If not specified, the workchain will run seekpath to generate "
                "a primitive cell and a bands_kpoints. Specify either this or `bands_kpoints_distance`."
            ),
        )
        spec.input(
            "bands_kpoints_distance",
            valid_type=orm.Float,
            serializer=orm.to_aiida_type,
            required=False,
            help="Minimum kpoints distance for seekpath to generate a list of kpoints along the path. "
            "Specify either this or `bands_kpoints`.",
        )
        spec.expose_inputs(
            YamboConvergence,
            namespace="yambo",
            exclude=("clean_workdir", "structure"),
            namespace_options={
                "help": "Inputs for the `YamboConvergence` for yambo calculation.",
                "required": False,
            },
        )
        spec.expose_inputs(
            Wannier90OptimizeWorkChain,
            namespace="wannier90",
            exclude=(
                "clean_workdir",
                "structure",
                "wannier90.wannier90.kpoint_path",
                "wannier90.wannier90.bands_kpoints",
            ),
            namespace_options={
                "help": "Inputs for the `Wannier90OptimizeWorkChain` for wannier90 calculation.",
                "required": False,
            },
        )
        spec.expose_inputs(
            YamboWorkflow,
            namespace="yambo_qp",
            exclude=("clean_workdir", "structure"),
            namespace_options={
                "help": (
                    "Inputs for the `YamboConvergence` for yambo QP calculation. "
                    "If not provided, it will be generated based on the previous converged inputs."
                ),
                "required": False,
            },
        )
        spec.expose_inputs(
            YppRestart,
            namespace="ypp",
            exclude=("clean_workdir", "structure"),
            namespace_options={
                "help": "Inputs for the `YppRestart` calculation. ",
                "required": False,
            },
        )
        spec.expose_inputs(
            Gw2wannier90Calculation,
            namespace="gw2wannier90",
            exclude=("clean_workdir", "structure"),
            namespace_options={
                "help": "Inputs for the `Gw2wannier90Calculation`. ",
                "required": False,
            },
        )
        spec.expose_inputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_qp",
            exclude=(
                "clean_workdir",
                "structure",
                "wannier90.kpoint_path",
                "wannier90.bands_kpoints",
            ),
            namespace_options={
                "help": (
                    "Inputs for the `Wannier90BaseWorkChain` for wannier90 QP calculation. "
                    "If not provided, it will be generated based on the previous wannier inputs."
                ),
                "required": False,
            },
        )

        spec.inputs.validator = validate_inputs

        spec.output(
            "primitive_structure",
            valid_type=orm.StructureData,
            required=False,
            help="The normalized and primitivized structure for which the calculations are computed.",
        )
        spec.output(
            "seekpath_parameters",
            valid_type=orm.Dict,
            required=False,
            help="The parameters used in the SeeKpath call to normalize the input or relaxed structure.",
        )
        spec.expose_outputs(
            YamboConvergence, namespace="yambo", namespace_options={"required": False}
        )
        spec.expose_outputs(
            Wannier90OptimizeWorkChain,
            namespace="wannier90",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            YamboWorkflow, namespace="yambo_qp", namespace_options={"required": False}
        )
        spec.expose_outputs(
            YppRestart, namespace="ypp", namespace_options={"required": False}
        )
        spec.expose_outputs(
            Gw2wannier90Calculation,
            namespace="gw2wannier90",
            namespace_options={"required": False},
        )
        spec.expose_outputs(Wannier90BaseWorkChain, namespace="wannier90_qp")

        spec.outline(
            cls.setup,
            if_(cls.should_run_seekpath)(
                cls.run_seekpath,
            ),
            if_(cls.should_run_yambo_convergence)(
                cls.run_yambo_convergence,
                cls.inspect_yambo_convergence,
            ),
            cls.setup_kmesh,
            if_(cls.should_run_wannier)(
                cls.run_wannier,
                cls.inspect_wannier,
            ),
            if_(cls.should_run_yambo_qp)(
                cls.run_yambo_qp,
                # use conv output folder, settins
                # builder.yres['yambo']['settings'] = Dict(dict={'COPY_DBS': True})
                cls.inspect_yambo_qp,
            ),
            # TODO run yambo on shifted grid
            if_(cls.should_run_ypp)(
                cls.run_ypp,
                cls.inspect_ypp,
            ),
            if_(cls.should_run_gw2wannier90)(
                cls.run_gw2wannier90,
                cls.inspect_gw2wannier90,
            ),
            cls.run_wannier_qp,
            cls.inspect_wannier_qp,
            cls.results,
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_SETUP",
            message="Unrecoverable error when running setup.",
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_YAMBO_CONV",
            message="Unrecoverable error when running yambo convergence.",
        )
        spec.exit_code(
            403,
            "ERROR_SUB_PROCESS_FAILED_SETUP_KMESH",
            message="Unrecoverable error when running setup_kmesh.",
        )
        spec.exit_code(
            404,
            "ERROR_SUB_PROCESS_FAILED_WANNIER",
            message="Unrecoverable error when running wannier.",
        )
        spec.exit_code(
            405,
            "ERROR_SUB_PROCESS_FAILED_YAMBO_QP",
            message="Unrecoverable error when running yambo QP correction.",
        )
        spec.exit_code(
            406,
            "ERROR_SUB_PROCESS_FAILED_YPP",
            message="Unrecoverable error when running yambo ypp.",
        )
        spec.exit_code(
            407,
            "ERROR_SUB_PROCESS_FAILED_GW2WANNIER90",
            message="Unrecoverable error when running gw2wannier90.",
        )
        spec.exit_code(
            408,
            "ERROR_SUB_PROCESS_FAILED_WANNIER_QP",
            message="Unrecoverable error when running wannier with QP-corrected eig.",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return the ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "yambo_wannier90.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=too-many-statements
        cls,
        codes: ty.Dict[str, ty.Union[orm.Code, str, int]],
        structure: orm.StructureData,
        *,
        protocol: str = None,
        overrides: dict = None,
        pseudo_family: str = "PseudoDojo/0.4/PBE/SR/standard/upf",
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param codes: [description]
        :type codes: typing.Dict[str, typing.Union[aiida.orm.Code, str, int]]
        :param bxsf: [description]
        :type bxsf: aiida.orm.RemoteData
        :param protocol: [description], defaults to None
        :type protocol: str, optional
        :param overrides: [description], defaults to None
        :type overrides: dict, optional
        :return: [description]
        :rtype: aiida.engine.ProcessBuilder
        """
        from aiida_wannier90_workflows.utils.workflows.builder import (
            recursive_merge_builder,
        )

        required_codes = [
            "pw",
            "pw2wannier90",
            "wannier90",
            "yambo",
            "p2y",
            "ypp",
            "gw2wannier90",
        ]
        if not all(_ in codes for _ in required_codes):
            raise ValueError(f"`codes` must contain {required_codes}")

        for key, code in codes.items():
            if not isinstance(code, orm.Code):
                codes[key] = orm.load_code(code)

        type_check(structure, orm.StructureData)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        builder = cls.get_builder()
        builder = recursive_merge_builder(builder, inputs)

        builder.structure = structure

        # Prepare yambo
        yambo_builder = get_builder_yambo(
            pw_code=codes["pw"],
            p2y_code=codes["p2y"],
            yambo_code=codes["yambo"],
            structure=structure,
            pseudo_family=pseudo_family,
        )
        inputs.yambo = yambo_builder._inputs(
            prune=True
        )  # pylint: disable=protected-access

        # Prepare wannier
        wannier_builder = Wannier90BandsWorkChain.get_builder_from_protocol(
            codes,
            structure,
            pseudo_family=pseudo_family,
            exclude_semicore=False,
        )
        inputs.wannier = wannier_builder._inputs(
            prune=True
        )  # pylint: disable=protected-access

        # Ypp
        ypp_builder = YppRestart.get_builder_from_protocol(
            code=codes["ypp"],
        )

        metadata = {
            "options": {
                # 'queue_name': 'debug',
                # 'account': 'mr0',
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "num_cores_per_mpiproc": 1,
                },
                "prepend_text": "export OMP_NUM_THREADS="
                + str(1)
                + "\nmv ./SAVE/ndb.QP* .",
                # 'max_wallclock_seconds': 60 * 5 * 1,
                "withmpi": True,
            },
        }
        ypp_builder.ypp.metadata = metadata
        ypp_builder.ypp.parameters = orm.Dict(
            dict={
                # 'arguments': ['infver', 'QPDBs', 'QPDB_merge'],
                # 'variables': {
                #     'BoseTemp': [0, 'eV'],
                # },
                "arguments": [
                    "wannier",
                ],
                "variables": {
                    "WriteAMU": "",
                },
            }
        )
        ypp_builder.ypp.QP_calculations = List(
            list=[1948, 1980, 2006, 2064, 2151, 2176, 2215, 2253]
        )
        ypp_builder.QP_DB = load_node(2329)

        builder = cls.get_builder()
        builder = recursive_merge_builder(builder, inputs)

        return builder

    def setup(self) -> None:
        """Initialize context variables."""

        self.ctx.current_structure = self.inputs.structure

        # Commensurate meshes for GW and W90
        self.ctx.kpoints_gw = None
        self.ctx.kpoints_w90 = None

        if self.should_run_yambo_convergence():
            self.ctx.kpoints_gw = self.inputs.yambo.nscf.kpoints
        else:
            # Then `yambo_qp` must be in the wkchain inputs
            parent_folder = self.inputs.yambo_qp.parent_folder
            # Creator is a YamboCalculation, caller is a YamboRestart
            wkchain_gw = parent_folder.creator.caller
            # Its parent_folder is the remote_folder of a pw.x nscf
            calc_nscf = wkchain_gw.inputs.parent_folder.creator
            self.ctx.kpoints_gw = calc_nscf.inputs.kpoints

        if self.should_run_wannier():
            self.ctx.kpoints_w90 = self.inputs.wannier90.nscf.kpoints
        else:
            self.ctx.kpoints_w90 = self.inputs.wannier90_qp.wannier90.kpoints

    def should_run_seekpath(self):
        """Run seekpath if the `inputs.bands_kpoints` is not provided."""
        return "bands_kpoints" not in self.inputs

    def run_seekpath(self):
        """Run the structure through SeeKpath to get the primitive and normalized structure."""

        args = {
            "structure": self.inputs.structure,
            "metadata": {"call_link_label": "seekpath_structure_analysis"},
        }
        if "bands_kpoints_distance" in self.inputs:
            args["reference_distance"] = self.inputs["bands_kpoints_distance"]

        result = seekpath_structure_analysis(**args)

        self.ctx.current_structure = result["primitive_structure"]
        self.ctx.current_bands_kpoints = result["explicit_kpoints"]

        structure_formula = self.inputs.structure.get_formula()
        primitive_structure_formula = result["primitive_structure"].get_formula()
        self.report(
            f"launching seekpath: {structure_formula} -> {primitive_structure_formula}"
        )

        self.out("primitive_structure", result["primitive_structure"])
        self.out("seekpath_parameters", result["parameters"])

    def should_run_yambo_convergence(self) -> bool:
        """Whether to run yambo convergence."""
        if "yambo" in self.inputs:
            return True

        return False

    def run_yambo_convergence(self) -> ty.Dict:
        """Run the `YamboConvergence`."""
        inputs = AttributeDict(self.exposed_inputs(YamboConvergence, namespace="yambo"))

        inputs.metadata.call_link_label = "yambo_convergence"

        inputs.ywfl.scf.pw.structure = self.ctx.current_structure
        inputs.ywfl.nscf.pw.structure = self.ctx.current_structure

        inputs = prepare_process_inputs(YamboConvergence, inputs)
        running = self.submit(YamboConvergence, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_yambo_conv=running)

    def inspect_yambo_convergence(self) -> ty.Union[None, ExitCode]:
        """Verify that the `Wan2skeafCalculation` successfully finished."""
        wkchain = self.ctx.wkchain_yambo_conv

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_YAMBO_CONV

        self.ctx.kpoints_gw = wkchain.inputs.nscf.kpoints

    def setup_kmesh(self) -> None:
        """Find commensurate kmeshes for both Yambo and wannier90."""
        kpoints_gw = self.ctx.kpoints_gw
        kpoints_w90 = self.ctx.kpoints_w90

        result = find_commensurate_meshes(
            dense_mesh=kpoints_gw, coarse_mesh=kpoints_w90
        )
        dense_mesh = result["dense_mesh"]
        coarse_mesh = result["coarse_mesh"]

        kmesh_gw = get_mesh_from_kpoints(kpoints_gw)
        kmesh_w90 = get_mesh_from_kpoints(kpoints_w90)

        self.report(
            f"Converged GW kmesh = {kmesh_gw}, W90 input kmesh = {kmesh_w90}. "
            f"Found commensurate meshes GW = {dense_mesh}, W90 = {coarse_mesh}."
        )

        if self.should_run_wannier():
            # Use theses meshes before submitting the corresponding workflow
            self.ctx.kpoints_w90 = get_explicit_kpoints_from_mesh(kmesh_w90)
        else:
            if not np.allclose(coarse_mesh, kmesh_w90):
                self.report(
                    f"The kmesh {kmesh_w90} of the input `wannier90_qp` is "
                    f"different from the commensurate kmesh {coarse_mesh}."
                )
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SETUP_KMESH

        if self.should_run_yambo_qp():
            # Use theses meshes before submitting the corresponding workflow
            self.ctx.kpoints_gw = create_kpoints_from_mesh(kmesh_gw)
        else:
            if not np.allclose(dense_mesh, kmesh_gw):
                self.report(
                    f"The kmesh {kmesh_gw} of the input `yambo_qp` is "
                    f"different from the commensurate kmesh {dense_mesh}."
                )
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SETUP_KMESH

    def should_run_wannier(self) -> bool:
        """Whether to run wannier."""
        if "wannier90" in self.inputs:
            return True

        return False

    def run_wannier(self) -> ty.Dict:
        """Run the `Wannier90BandsWorkChain`."""
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace="wannier90")
        )
        inputs.metadata.call_link_label = "wannier90"

        # Use commensurate kmesh
        inputs.nscf.kpoints = self.ctx.kpoints_w90
        inputs.wannier90.wannier90.kpoints = self.ctx.kpoints_w90

        inputs = prepare_process_inputs(Wannier90OptimizeWorkChain, inputs)
        running = self.submit(Wannier90OptimizeWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_wannier90=running)

    def inspect_wannier(self) -> ty.Union[None, ExitCode]:
        """Verify that the `Wannier90BandsWorkChain` successfully finished."""
        wkchain = self.ctx.wkchain_wannier90

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER

    def should_run_yambo_qp(self) -> bool:
        """Whether to run yambo_qp."""
        if "yambo_qp" in self.inputs:
            return True

        if "yambo" in self.inputs:
            return True

        return False

    def prepare_yambo_qp_inputs(self) -> AttributeDict:
        """Prepare inputs for yambo QP."""
        # Get the converged input from YamboWorkflow
        if "yambo_qp" in self.inputs:
            inputs = AttributeDict(
                self.exposed_inputs(YamboWorkflow, namespace="yambo_qp")
            )
        else:
            yambo_wkchain = self.ctx.wkchain_yambo_conv

            # Find converged wkchain inputs
            yambo_history = yambo_wkchain.outputs.history.get_dict()
            # e.g.
            #  'useful': {'0': False, '1': False, '2': False, '3': False, '4': True,
            #             '5': False, '6': False, '7': False, '8': False, '9': False}
            converged_wkchain = [k for k, v in yambo_history["useful"].items() if v][0]
            # The YamboWorkflow
            converged_wkchain = orm.load_node(yambo_history["uuid"][converged_wkchain])

            inputs = converged_wkchain.inputs._construct_attribute_dict(True)

        # Prepare QPkrange
        if self.should_run_wannier():
            w90_calc_inputs = self.ctx.wkchain_wannier90.inputs.wannier90.wannier90
        else:
            w90_calc_inputs = self.inputs.wannier90_qp.wannier90
        w90_params = w90_calc_inputs.parameters.get_dict()

        num_bands = w90_params["num_bands"]
        exclude_bands = w90_params.get("exclude_bands", [0])
        start_band = max(exclude_bands) + 1
        end_band = start_band + num_bands - 1

        p2y_nscf_wkchain = (
            converged_wkchain.get_outgoing(
                link_label_filter="nscf",
            )
            .one()
            .node
        )

        gw_kpoints = get_output_explicit_kpoints(p2y_nscf_wkchain.outputs.retrieved)
        qpkrange = kmapper(
            dense_mesh=gw_kpoints,
            coarse_mesh=self.ctx.kpoints_w90,
            start_band=orm.Int(start_band),
            end_band=orm.Int(end_band),
        )
        qpkrange = qpkrange.get_list()

        # Set QPkrange in GW parameters
        yambo_params = inputs.parameters.get_dict()
        # TODO what is the empty string?
        yambo_params["variables"]["QPkrange"] = [qpkrange, ""]

        inputs.parameters = orm.Dict(dict=yambo_params)

        return inputs

    def run_yambo_qp(self) -> ty.Dict:
        """Run the `Wannier90BandsWorkChain`."""
        inputs = self.prepare_yambo_qp_inputs()

        inputs.metadata.call_link_label = "yambo_qp"
        inputs = prepare_process_inputs(YamboWorkflow, inputs)
        running = self.submit(YamboWorkflow, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_yambo_qp=running)

    def inspect_yambo_qp(self) -> ty.Union[None, ExitCode]:
        """Verify that the `YamboWorkflow` successfully finished."""
        wkchain = self.ctx.wkchain_yambo_qp

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_YAMBO_QP

    def should_run_ypp(self) -> bool:
        """Whether to run ypp."""
        if "ypp" in self.inputs:
            return True

        return False

    def prepare_ypp_inputs(self) -> AttributeDict:
        """Prepare inputs for ypp."""
        inputs = AttributeDict(
            self.exposed_inputs(YamboConvergence, namespace="yambo")
        ).yres

        if self.should_run_yambo_convergence():
            inputs.parent_folder = self.ctx.wkchain_yambo_conv.outputs.remote_folder

        if "nnkp_file" not in inputs:
            inputs.nnkp_file = self.ctx.wkchain_wannier90.outputs.wannier90_pp.nnkp

        return inputs

    def run_ypp(self) -> ty.Dict:
        """Run the `ypp`."""
        inputs = self.prepare_ypp_inputs()

        inputs.metadata.call_link_label = "ypp"
        inputs = prepare_process_inputs(YppRestart, inputs)
        running = self.submit(YppRestart, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_ypp=running)

    def inspect_ypp(self) -> ty.Union[None, ExitCode]:
        """Verify that the `YamboWorkflow` successfully finished."""
        wkchain = self.ctx.wkchain_ypp

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_YPP

    def should_run_gw2wannier90(self) -> bool:
        """Whether to run gw2wannier90."""
        if "gw2wannier90" in self.inputs:
            return True

        return False

    def run_gw2wannier90(self) -> ty.Dict:
        """Run the ``gw2wannier90``."""
        inputs = AttributeDict(
            self.exposed_inputs(Gw2wannier90Calculation, namespace="gw2wannier90")
        )
        inputs.metadata.call_link_label = "gw2wannier90"

        if self.should_run_wannier():
            inputs.nnkp = self.ctx.wkchain_wannier90.outputs.wannier90_pp.nnkp

        if self.should_run_ypp():
            inputs.unsorted_eig = self.ctx.wkchain_ypp.outputs.unsorted_eig_file

        inputs = prepare_process_inputs(Gw2wannier90Calculation, inputs)
        running = self.submit(Gw2wannier90Calculation, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(calc_gw2wannier90=running)

    def inspect_gw2wannier90(self) -> ty.Union[None, ExitCode]:
        """Verify that the `Gw2wannier90Calculation` successfully finished."""
        calc = self.ctx.calc_gw2wannier90

        if not calc.is_finished_ok:
            self.report(
                f"{calc.process_label} failed with exit status {calc.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GW2WANNIER90

    def run_wannier_qp(self) -> ty.Dict:
        """Run the `wannier_qp`."""
        if "wannier90_qp" in self.inputs:
            inputs = AttributeDict(
                self.exposed_inputs(Wannier90BaseWorkChain, namespace="wannier90_qp")
            )
        else:
            wannier_wkchain = self.ctx.wkchain_wannier90
            # TODO shift_energy_windows, use fixed inputs.parameters
            inputs = wannier_wkchain.inputs._construct_attribute_dict(True)

        inputs.metadata.call_link_label = "wannier90_qp"
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_wannier90_qp=running)

    def inspect_wannier_qp(self) -> ty.Union[None, ExitCode]:
        """Verify that the `Wannier90BaseWorkChain` successfully finished."""
        wkchain = self.ctx.wkchain_wannier90_qp

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER_QP

    def results(self) -> None:
        """Attach the relevant output nodes."""

        if "wkchain_yambo_conv" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.wkchain_yambo_conv,
                    YamboConvergence,
                    namespace="yambo",
                )
            )

        if "wkchain_wannier90" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.wkchain_wannier90,
                    Wannier90OptimizeWorkChain,
                    namespace="wannier90",
                )
            )

        if "wkchain_yambo_qp" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.wkchain_yambo_qp,
                    YamboWorkflow,
                    namespace="yambo_qp",
                )
            )

        if "wkchain_ypp" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.wkchain_ypp,
                    YppRestart,
                    namespace="ypp",
                )
            )

        if "calc_gw2wannier90" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.calc_gw2wannier90,
                    Gw2wannier90Calculation,
                    namespace="gw2wannier90",
                )
            )

        self.out_many(
            self.exposed_outputs(
                self.ctx.wkchain_wannier90_qp,
                Wannier90BaseWorkChain,
                namespace="wannier90_qp",
            )
        )

        self.report(f"{self.get_name()} successfully completed")

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if not self.inputs.clean_workdir:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )


def get_builder_yambo(
    pw_code: orm.Code,
    p2y_code: orm.Code,
    yambo_code: orm.Code,
    structure: orm.StructureData,
    pseudo_family: str,
) -> ProcessBuilder:

    # TODO yamboworkflow run seekpath -> primitive cell?
    metadata_pw = {
        "options": {
            # 'queue_name':'s3par',
            "resources": {
                "num_machines": 1,
                # 'num_mpiprocs_per_machine': 8,
                "num_cores_per_mpiproc": 1,
            },
            "prepend_text": "export OMP_NUM_THREADS=" + str(1),
            "max_wallclock_seconds": 43200,
            "withmpi": True,
        },
    }

    ecutwfc = 80
    overrides_scf = {
        "pseudo_family": pseudo_family,
        "pw": {
            "parameters": {
                "SYSTEM": {
                    "ecutwfc": ecutwfc,
                    "force_symmorphic": True,
                },
            },
            "metadata": metadata_pw,
        },
    }

    overrides_nscf = {
        "pseudo_family": pseudo_family,
        "pw": {
            "parameters": {
                "CONTROL": {
                    "calculation": "nscf",
                },
                "SYSTEM": {
                    "ecutwfc": ecutwfc,
                    "force_symmorphic": True,
                },
            },
            "metadata": metadata_pw,
        },
    }

    overrides_yambo = {
        "clean_workdir": False,
        "metadata": {
            "options": {
                # 'queue_name':'s3par',
                "resources": {
                    "num_machines": 1,
                    # 'num_mpiprocs_per_machine': 8,
                    "num_cores_per_mpiproc": 1,
                },
                "prepend_text": "export OMP_NUM_THREADS=" + str(1),
                "max_wallclock_seconds": 60 * 60 * 1,
                "withmpi": True,
            },
        },
        "yambo": {
            "parameters": {
                "arguments": [
                    "NLCC",
                    "rim_cut",
                ],
                "variables": {
                    "NGsBlkXp": [4, "Ry"],
                    "BndsRnXp": [[1, 200], ""],
                    "GbndRnge": [[1, 200], ""],
                    "RandQpts": [5000000, ""],
                    "RandGvec": [100, "RL"],
                    #'X_and_IO_CPU' : '1 1 1 32 1',
                    #'X_and_IO_ROLEs' : 'q k g c v',
                    #'DIP_CPU' :'1 32 1',
                    #'DIP_ROLEs' : 'k c v',
                    #'SE_CPU' : '1 1 32',
                    #'SE_ROLEs' : 'q qp b',
                    "QPkrange": [[[1, 1, 32, 32]], ""],
                },
            },
        },
    }

    overrides = {
        "ywfl": {
            "scf": overrides_scf,
            "nscf": overrides_nscf,
            "yres": overrides_yambo,
        }
    }

    yambo_builder = YamboConvergence.get_builder_from_protocol(
        pw_code=pw_code,
        p2y_code=p2y_code,
        code=yambo_code,
        protocol="moderate",
        structure=structure,
        overrides=overrides,
        # parent_folder=load_node(225176).outputs.remote_folder,
    )

    return yambo_builder
