#!/usr/bin/env python
"""Base class for Yambo+Wannier90 workflow."""
from email.charset import QP
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
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)
from aiida_wannier90_workflows.utils.kpoints import (
    get_explicit_kpoints,
    get_mesh_from_kpoints,
)
from aiida_wannier90_workflows.utils.workflows.builder import set_kpoints
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90BaseWorkChain,
    Wannier90OptimizeWorkChain,
)

from aiida_yambo.workflows.yamboconvergence import YamboConvergence
from aiida_yambo.workflows.yamborestart import YamboRestart
from aiida_yambo.workflows.yambowf import YamboWorkflow
from aiida_yambo.workflows.ypprestart import YppRestart

from aiida_yambo_wannier90.calculations.functions.kmesh import (
    find_commensurate_meshes,
    get_output_explicit_kpoints,
    is_commensurate,
    kmapper,
)
from aiida_yambo_wannier90.calculations.gw2wannier90 import Gw2wannier90Calculation
from aiida_yambo_wannier90.common.types import Gw2wannier90SortMode
from aiida_yambo_wannier90.utils.workflows import (
    get_yambo_converged_workchain,
    get_yambo_nscf,
)

__all__ = ["validate_inputs", "YamboWannier90WorkChain"]


# pylint: disable=too-many-lines
# pylint: disable=fixme
# TODO remove this todo disable


def validate_inputs(  # pylint: disable=inconsistent-return-statements,too-many-return-statements,too-many-branches,too-many-locals
    inputs: dict, ctx=None  # pylint: disable=unused-argument
) -> ty.Union[None, str]:
    """Validate the inputs of the entire input namespace."""

    # Must run steps sequentially
    order = ["yambo", "yambo_qp", "ypp", "wannier90", "gw2wannier90", "wannier90_qp"]
    non_empty = [_ in inputs for _ in order]
    first_input = non_empty.index(True)
    if not all(non_empty[first_input:]):
        first_no_input = first_input + non_empty[first_input:].index(False)
        return (
            f"WorkChain must be run in order, `{order[first_input]}` is provided "
            f"but `{order[first_no_input]}` is empty."
        )

    # Check inputs if previous steps are skipped
    should_run_yambo = "yambo" in inputs
    should_run_yambo_commensurate = "GW_mesh" in inputs
    should_run_wannier90 = "wannier90" in inputs
    should_run_yambo_qp = "yambo_qp" in inputs
    should_run_ypp = "ypp" in inputs
    should_run_gw2wannier90 = "gw2wannier90" in inputs
    should_run_wannier90_qp = "wannier90_qp" in inputs

    if should_run_yambo_qp:
        yambo_qp_inputs = inputs["yambo_qp"]

        if not should_run_yambo:
            if "parent_folder" not in yambo_qp_inputs and not should_run_yambo_commensurate:
                return "`yambo_qp.parent_folder` is empty."

    if should_run_ypp:
        ypp_inputs = inputs["ypp"]

        if not should_run_yambo_qp:
            if "QP_DB" not in ypp_inputs["ypp"]:
                return "`ypp.ypp.QP_DB` is empty."
            if "parent_folder" not in ypp_inputs:
                return "`ypp.parent_folder` is empty."

        # I need `wannier90` input to run a w90 postproc before `ypp`,
        # or if there is `nnkp`, I skip the postproc.
        if not should_run_wannier90:
            if "nnkp_file" not in ypp_inputs["ypp"]:
                return "`ypp.ypp.nnkp_file` is empty."

    if should_run_gw2wannier90:
        gw2wannier90_inputs = inputs["gw2wannier90"]
        if not should_run_wannier90:
            for tag in ["nnkp", "parent_folder"]:
                if tag not in gw2wannier90_inputs:
                    return f"`gw2wannier90.{tag}` is empty."

        if not should_run_ypp:
            if "unsorted_eig" not in gw2wannier90_inputs:
                return "`gw2wannier90.unsorted_eig` is empty."

    if should_run_wannier90_qp:
        wannier90_qp_inputs = inputs["wannier90_qp"]
        if not should_run_gw2wannier90:
            if "remote_input_folder" not in wannier90_qp_inputs["wannier90"]:
                return "`wannier90_qp.wannier90.remote_input_folder` is empty."


class YamboWannier90WorkChain(
    ProtocolMixin, WorkChain
):  # pylint: disable=too-many-public-methods
    """Workchain to obtain GW-corrected maximally localised Wannier functions (MLWF)."""

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        from aiida_wannier90_workflows.workflows.base.wannier90 import (
            validate_inputs_base as validate_inputs_base_wannier90,
        )

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
        spec.input(
            "kpoints_force_gw",
            valid_type=orm.Bool,
            serializer=orm.to_aiida_type,
            default=lambda: orm.Bool(False),
            help="If `True` will force W90 to use the GW converged k-point mesh.",
        )
        spec.input(
            "GW_mesh",
            valid_type=orm.KpointsData,
            serializer=orm.to_aiida_type,
            required=False,
            help="GW mesh. This allow to start from yambo commensurate, skipping gw convergence",
        )
        spec.expose_inputs(
            YamboConvergence,
            namespace="yambo",
            exclude=(
                "clean_workdir",
                "ywfl.scf.pw.structure",
                "ywfl.nscf.pw.structure",
            ),
            namespace_options={
                "help": "Inputs for the `YamboConvergence` for yambo calculation.",
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.expose_inputs(
            YamboWorkflow,
            namespace="yambo_qp",
            exclude=(
                "clean_workdir",
                "scf.pw.structure",
                "nscf.pw.structure",
            ),
            namespace_options={
                "help": (
                    "Inputs for the `YamboConvergence` for yambo QP calculation. "
                    "If not provided, it will be generated based on the previous converged inputs."
                ),
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.expose_inputs(
            YppRestart,
            namespace="ypp",
            exclude=("clean_workdir",),
            namespace_options={
                "help": "Inputs for the `YppRestart` calculation, to be used for unsorted.eig generation. ",
                "required": False,
                "populate_defaults": False,
            },
        )

        spec.expose_inputs(
            YppRestart,
            namespace="ypp_QP",
            exclude=("clean_workdir",),
            namespace_options={
                "help": "Inputs for the `YppRestart` calculation, to be used for merging QP dbs. ",
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.expose_inputs(
            Wannier90OptimizeWorkChain,
            namespace="wannier90",
            exclude=(
                "clean_workdir",
                "structure",
                "kpoint_path",
                "bands_kpoints",
                "bands_kpoints_distance",
            ),
            namespace_options={
                "help": "Inputs for the `Wannier90OptimizeWorkChain` for wannier90 calculation.",
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.expose_inputs(
            Gw2wannier90Calculation,
            namespace="gw2wannier90",
            exclude=("clean_workdir",),
            namespace_options={
                "help": "Inputs for the `Gw2wannier90Calculation`. ",
                "required": False,
                "populate_defaults": False,
            },
        )
        spec.expose_inputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_qp",
            exclude=(
                "clean_workdir",
                "wannier90.structure",
                "wannier90.kpoint_path",
                "wannier90.bands_kpoints",
            ),
            namespace_options={
                "help": (
                    "Inputs for the `Wannier90BaseWorkChain` for wannier90 QP calculation. "
                    "If not provided, it will be generated based on the previous wannier inputs."
                ),
                "required": True,
            },
        )
        spec.inputs["wannier90_qp"].validator = validate_inputs_base_wannier90

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
            YamboConvergence,
            namespace="yambo",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            YamboWorkflow,
            namespace="yambo_commensurate",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            YamboWorkflow,
            namespace="yambo_qp",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_pp",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            YppRestart,
            namespace="ypp",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90OptimizeWorkChain,
            namespace="wannier90",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Gw2wannier90Calculation,
            namespace="gw2wannier90",
            namespace_options={"required": False},
        )
        spec.expose_outputs(
            Wannier90BaseWorkChain,
            namespace="wannier90_qp",
        )
        spec.output(
            "band_structures.wannier90",
            valid_type=orm.BandsData,
            required=False,
            help="The Wannier interpolated band structure at DFT level.",
        )
        spec.output(
            "band_structures.wannier90_qp",
            valid_type=orm.BandsData,
            help="The Wannier interpolated band structure at G0W0 level.",
        )

        spec.outline(
            cls.setup,
            if_(cls.should_run_seekpath)(
                cls.run_seekpath,
            ),
            if_(cls.should_run_yambo_convergence)(
                cls.run_yambo_convergence,
                cls.inspect_yambo_convergence,
            ),
            if_(cls.should_run_setup_kmesh)(
                cls.setup_kmesh,
            ),
            if_(cls.should_run_yambo_commensurate)(
                cls.run_yambo_commensurate,
                cls.inspect_yambo_commensurate,
            ),
            # TODO run an additional yambo_qp on shifted grid to check w90_qp bands
            if_(cls.should_run_yambo_qp)(
                cls.run_yambo_qp,
                cls.inspect_yambo_qp,
            ),
            # if_(cls.should_run_ypp_qp)(
            #    cls.run_ypp_qp,
            #    cls.inspect_ypp_qp,
            # ),
            if_(cls.should_run_wannier90_pp)(
                cls.run_wannier90_pp,
                cls.inspect_wannier90_pp,
            ),
            if_(cls.should_run_ypp)(
                cls.run_ypp,
                cls.inspect_ypp,
            ),
            if_(cls.should_run_wannier90)(
                cls.run_wannier90,
                cls.inspect_wannier90,
            ),
            if_(cls.should_run_gw2wannier90)(
                cls.run_gw2wannier90,
                cls.inspect_gw2wannier90,
            ),
            cls.run_wannier90_qp,
            cls.inspect_wannier90_qp,
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
            "ERROR_SUB_PROCESS_FAILED_WANNIER90_PP",
            message="Unrecoverable error when running wannier90 postproc.",
        )
        spec.exit_code(
            405,
            "ERROR_SUB_PROCESS_FAILED_YAMBO_COMMENSURATE",
            message="Unrecoverable error when running yambo on commensurate kmesh.",
        )
        spec.exit_code(
            406,
            "ERROR_SUB_PROCESS_FAILED_YAMBO_QP",
            message="Unrecoverable error when running yambo QP correction.",
        )
        spec.exit_code(
            407,
            "ERROR_SUB_PROCESS_FAILED_YPP",
            message="Unrecoverable error when running yambo ypp.",
        )
        spec.exit_code(
            408,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90",
            message="Unrecoverable error when running wannier90.",
        )
        spec.exit_code(
            409,
            "ERROR_SUB_PROCESS_FAILED_GW2WANNIER90",
            message="Unrecoverable error when running gw2wannier90.",
        )
        spec.exit_code(
            410,
            "ERROR_SUB_PROCESS_FAILED_WANNIER90_QP",
            message="Unrecoverable error when running wannier90 with QP-corrected eig.",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return the ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        # pylint: disable=import-outside-toplevel
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "yambo_wannier90.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=too-many-statements,too-many-locals
        cls,
        codes: ty.Dict[str, ty.Union[orm.Code, str, int]],
        structure: orm.StructureData,
        *,
        protocol: str = None,
        overrides: dict = None,
        pseudo_family: str = "PseudoDojo/0.4/PBE/SR/standard/upf",
        exclude_semicore: bool = False,
        electronic_type=ElectronicType.METAL,
        wannier_projection_type: WannierProjectionType = WannierProjectionType.ATOMIC_PROJECTORS_QE,
        NLCC: bool = True,
        RIM_v: bool = True,
        RIM_W: bool = False,
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
        # pylint: disable=import-outside-toplevel,protected-access
        # from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
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

        inputs["structure"] = structure

        # Prepare yambo
        yambo_overrides = {
            "ywfl": {
                "scf": {"pseudo_family": pseudo_family},
                "nscf": {"pseudo_family": pseudo_family},
            },
        }
        yambo_builder = YamboConvergence.get_builder_from_protocol(
            pw_code=codes["pw"],
            preprocessing_code=codes["p2y"],
            code=codes["yambo"],
            protocol="moderate",
            structure=structure,
            electronic_type=electronic_type,
            overrides=yambo_overrides,
            NLCC=NLCC,
            RIM_v=RIM_v,
            RIM_W=RIM_W,
        )
        inputs["yambo"] = yambo_builder._inputs(prune=True)
        inputs["yambo"]["ywfl"]["scf"]["pw"].pop("structure", None)
        inputs["yambo"]["ywfl"]["nscf"]["pw"].pop("structure", None)
        inputs["yambo"].pop("clean_workdir", None)

        # Prepare wannier
        # projection_type = WannierProjectionType.ATOMIC_PROJECTORS_QE
        # disentanglement_type = WannierDisentanglementType.SMV
        # frozen_type = WannierFrozenType.FIXED_PLUS_PROJECTABILITY
        # Auto guess from projection_type
        disentanglement_type = None
        frozen_type = None
        wannier_builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            codes,
            structure,
            pseudo_family=pseudo_family,
            exclude_semicore=exclude_semicore,
            projection_type=wannier_projection_type,
            disentanglement_type=disentanglement_type,
            frozen_type=frozen_type,
        )
        # No reference PW bands, so we stop optimization
        wannier_builder.optimize_disproj = False
        inputs["wannier90"] = wannier_builder._inputs(prune=True)
        inputs["wannier90"].pop("structure", None)
        inputs["wannier90"].pop("clean_workdir", None)

        # TODO Prepare yambo_qp
        # yambo_qp_builder = YamboRestart.get_builder_from_protocol(
        #     pw_code=codes["pw"],
        #     preprocessing_code=codes["p2y"],
        #     code=codes["yambo"],
        #     protocol="moderate",
        #     NLCC=NLCC,
        #     RIM_v=RIM_v,
        #     RIM_W=RIM_W,
        # )
        # inputs["yambo_qp"] = yambo_qp_builder._inputs(prune=True)
        inputs["yambo_qp"] = inputs["yambo"]["ywfl"]
        inputs["yambo_qp"].pop("clean_workdir", None)

        # Ypp; without a parent_folder for now. We should set it during the input preparation
        ypp_builder = YppRestart.get_builder_from_protocol(
            code=codes["ypp"],
            protocol="Wannier",
        )
        # ypp_builder.ypp.QP_calculations = List(
        #    list=[1948, 1980, 2006, 2064, 2151, 2176, 2215, 2253]
        # )
        # ypp_builder.QP_DB = load_node(2329)
        inputs["ypp"] = ypp_builder._inputs(prune=True)
        inputs["ypp"].pop("clean_workdir", None)

        # ypp_QP
        ypp_builder = YppRestart.get_builder_from_protocol(
            code=codes["ypp"],
            protocol="merge_QP",
        )

        inputs["ypp_QP"] = ypp_builder._inputs(prune=True)
        inputs["ypp_QP"].pop(
            "clean_workdir", None
        )  # but actually I want to clean the wdir

        # Prepare gw2wannier90
        inputs["gw2wannier90"] = {
            "code": codes["gw2wannier90"],
        }

        # Prepare wannier90_qp
        wannier90_qp_builder = Wannier90BaseWorkChain.get_builder_from_protocol(
            code=codes["wannier90"],
            structure=structure,
            pseudo_family=pseudo_family,
            overrides={
                "meta_parameters": {
                    "exclude_semicore": exclude_semicore,
                }
            },
            electronic_type=electronic_type,
        )
        params = wannier90_qp_builder.wannier90.parameters.get_dict()
        params["bands_plot"] = True
        wannier90_qp_builder.wannier90.parameters = orm.Dict(dict=params)
        inputs["wannier90_qp"] = wannier90_qp_builder._inputs(prune=True)
        inputs["wannier90_qp"]["wannier90"].pop("structure", None)
        inputs["wannier90_qp"].pop("clean_workdir", None)

        builder = cls.get_builder()
        builder = recursive_merge_builder(builder, inputs)

        return builder

    def setup(self) -> None:  # pylint: disable=inconsistent-return-statements
        """Initialize context variables."""

        self.ctx.current_structure = self.inputs.structure

        if "bands_kpoints" in self.inputs:
            self.ctx.current_bands_kpoints = self.inputs.bands_kpoints

        # Converged mesh from YamboConvergence
        self.ctx.kpoints_gw_conv = None
        if self.should_run_setup_kmesh() and not self.should_run_yambo_convergence() and not "GW_mesh" in self.inputs:
            # `setup_kmesh` need `self.ctx.kpoints_gw_conv`, I assume that
            # the parent of `yambo_qp` is a converged mesh.
            # Since the workchain runs sequentially, the `yambo_qp` must be
            # in the workchain inputs.
            if "yambo_qp" in self.inputs:
                parent_folder = self.inputs.yambo_qp.parent_folder
            elif "ypp" in self.inputs:
                parent_folder = self.inputs.ypp.parent_folder
            # The creator is a YamboCalculation, caller is a YamboRestart
            wkchain_gw = parent_folder.creator.caller
            # Its parent_folder is the remote_folder of a pw.x nscf
            calc_nscf = wkchain_gw.inputs.parent_folder.creator
            self.ctx.kpoints_gw_conv = calc_nscf.inputs.kpoints

        # Input Wannier90 mesh
        self.ctx.kpoints_w90_input = None
        if self.should_run_wannier90():
            self.ctx.kpoints_w90_input = self.inputs.wannier90.nscf.kpoints

            if (
                not self.should_run_yambo_convergence()
                and not self.inputs.kpoints_force_gw
                # If starting wannier90+gw2wannier90+wannier90_qp from unsorted.eig,
                # then I don't know the gw converged mesh
                and self.ctx.kpoints_gw_conv is not None
            ):
                kmesh_gw_conv = get_mesh_from_kpoints(self.ctx.kpoints_gw_conv)
                kmesh_w90_input = get_mesh_from_kpoints(self.ctx.kpoints_w90_input)

                if not is_commensurate(kmesh_gw_conv, kmesh_w90_input) and not 'GW_mesh' in self.inputs:
                    self.report(
                        f"Skipping GW convergence, but GW converged mesh {kmesh_gw_conv} "
                        f"is not commensurate with W90 input mesh {kmesh_w90_input}"
                    )
                    return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SETUP

        # Commensurate meshes for GW and W90
        self.ctx.kpoints_gw = None
        # Initialize with input mesh
        self.ctx.kpoints_w90 = self.ctx.kpoints_w90_input

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

    def prepare_yambo_convergence_inputs(self) -> AttributeDict:
        """Prepare inputs for ``YamboConvergence``."""
        inputs = AttributeDict(self.exposed_inputs(YamboConvergence, namespace="yambo"))

        inputs.ywfl.scf.pw.structure = self.ctx.current_structure
        inputs.ywfl.nscf.pw.structure = self.ctx.current_structure

        return inputs

    def run_yambo_convergence(self) -> ty.Dict:
        """Run the ``YamboConvergence``."""
        inputs = self.prepare_yambo_convergence_inputs()

        inputs.metadata.call_link_label = "yambo_convergence"
        inputs = prepare_process_inputs(YamboConvergence, inputs)
        running = self.submit(YamboConvergence, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_yambo_conv=running)

    def inspect_yambo_convergence(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the `Wan2skeafCalculation` successfully finished."""
        wkchain = self.ctx.wkchain_yambo_conv

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_YAMBO_CONV

        # Find the converged kmesh
        converged_wkchain = get_yambo_converged_workchain(wkchain)
        nscf_wkchain = get_yambo_nscf(converged_wkchain)

        self.ctx.kpoints_gw_conv = nscf_wkchain.inputs.kpoints

    def should_run_setup_kmesh(self) -> bool:
        """Whether to run setup_kmesh."""
        if "GW_mesh" in self.inputs:
            self.ctx.kpoints_gw_conv = self.inputs.GW_mesh
        return self.should_run_yambo_convergence() or self.should_run_wannier90_pp()

    def setup_kmesh(self) -> None:
        """Find commensurate kmeshes for both Yambo and Wannier90."""
        kpoints_gw_conv = self.ctx.kpoints_gw_conv
        kpoints_w90_input = self.ctx.kpoints_w90_input

        kmesh_gw_conv = get_mesh_from_kpoints(kpoints_gw_conv)
        kmesh_w90_input = get_mesh_from_kpoints(kpoints_w90_input)

        if self.inputs.kpoints_force_gw:
            self.ctx.kpoints_gw = kpoints_gw_conv
            self.ctx.kpoints_w90 = get_explicit_kpoints(kpoints_gw_conv)
            self.report(
                f"Converged GW kmesh = {kmesh_gw_conv}, W90 input kmesh = {kmesh_w90_input}. "
                f"Force W90 using GW kmesh = {kmesh_gw_conv}."
            )
            return

        result = find_commensurate_meshes(  # pylint: disable=unexpected-keyword-arg
            dense_mesh=kpoints_gw_conv,
            coarse_mesh=kpoints_w90_input,
            metadata={"call_link_label": "find_commensurate_meshes"},
        )
        kpoints_dense = result["dense_mesh"]
        kpoints_coarse = result["coarse_mesh"]

        kmesh_dense = get_mesh_from_kpoints(kpoints_dense)
        kmesh_coarse = get_mesh_from_kpoints(kpoints_coarse)

        self.report(
            f"Converged GW kmesh = {kmesh_gw_conv}, W90 input kmesh = {kmesh_w90_input}. "
            f"Found commensurate meshes GW = {kmesh_dense}, W90 = {kmesh_coarse}."
        )

        # Use theses meshes before submitting the corresponding workflow
        if np.allclose(kmesh_coarse, kmesh_w90_input):
            self.ctx.kpoints_w90 = kpoints_w90_input
        else:
            self.ctx.kpoints_w90 = get_explicit_kpoints(kpoints_coarse)

        if np.allclose(kmesh_dense, kmesh_gw_conv):
            self.ctx.kpoints_gw = kpoints_gw_conv
        else:
            self.ctx.kpoints_gw = kpoints_dense

    def should_run_yambo_commensurate(self) -> bool:
        """Whether to run again yambo on the commensurate kmesh."""

        if "GW_mesh" in self.inputs and not 'parent_folder' in self.inputs["yambo_qp"]:
            return True

        if not self.should_run_yambo_convergence():
            return False

        if self.ctx.kpoints_gw_conv != self.ctx.kpoints_gw:
            return True

        return False

    def prepare_yambo_commensurate_inputs(self) -> AttributeDict:
        """Prepare inputs for yambo commensurate."""
        # Get and reuse the converged input from YamboWorkflow
        # pylint: disable=protected-access
        inputs = AttributeDict(self.exposed_inputs(YamboWorkflow, namespace="yambo_qp"))
        if "QP_subset_dict" in inputs:
            del inputs.QP_subset_dict
        inputs.scf.pw.structure = self.ctx.current_structure
        inputs.nscf.pw.structure = self.ctx.current_structure

        if self.should_run_yambo_convergence():
            converged_wkchain = get_yambo_converged_workchain(self.ctx.wkchain_yambo_conv)
            inputs.yres.yambo.parameters = converged_wkchain.inputs._construct_attribute_dict(True)

        # Use commensurate mesh
        inputs.nscf.kpoints = self.ctx.kpoints_gw

        # Set parallelization, mpi_procs, npool, ...
        # `inputs.yambo_qp` always exists, but `inputs.yambo` might be empty
        if "scf" in inputs and "scf" in self.inputs.yambo_qp:
            inputs.scf.pw.metadata = self.inputs.yambo_qp.scf.pw.metadata
            if "parallelization" in self.inputs.yambo_qp.scf.pw:
                inputs.scf.pw.parallelization = (
                    self.inputs.yambo_qp.scf.pw.parallelization
                )
        if "pw" in inputs.nscf and "pw" in self.inputs.yambo_qp.nscf:
            inputs.nscf.pw.metadata = self.inputs.yambo_qp.nscf.pw.metadata
            if "parallelization" in self.inputs.yambo_qp.nscf.pw:
                inputs.nscf.pw.parallelization = (
                    self.inputs.yambo_qp.nscf.pw.parallelization
                )

        return inputs

    def run_yambo_commensurate(self) -> ty.Dict:
        """Run the `YamboWorkflow`."""
        inputs = self.prepare_yambo_commensurate_inputs()

        inputs.metadata.call_link_label = "yambo_commensurate"
        inputs = prepare_process_inputs(YamboWorkflow, inputs)
        running = self.submit(YamboWorkflow, **inputs)
        self.report(
            f"launching {running.process_label}<{running.pk}> for yambo_commensurate"
        )

        return ToContext(wkchain_yambo_commensurate=running)

    def inspect_yambo_commensurate(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the `YamboWorkflow` successfully finished."""
        wkchain = self.ctx.wkchain_yambo_commensurate

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_YAMBO_COMMENSURATE

    def should_run_yambo_qp(self) -> bool:
        """Whether to run yambo_qp."""
        if "yambo_qp" in self.inputs:
            return True

        return False

    def prepare_yambo_qp_inputs(self) -> AttributeDict:
        """Prepare inputs for yambo QP."""
        # pylint: disable=too-many-locals
        # Get the converged input from YamboWorkflow
        inputs = AttributeDict(self.exposed_inputs(YamboWorkflow, namespace="yambo_qp"))
        yambo_params = inputs.yres.yambo.parameters.get_dict()

        # Prepare QPkrange
        if self.should_run_wannier90():
            # w90_calc_inputs = self.ctx.wkchain_wannier90.inputs.wannier90.wannier90
            w90_calc_inputs = self.inputs.wannier90.wannier90.wannier90
        else:
            w90_calc_inputs = self.inputs.wannier90_qp.wannier90
        w90_params = w90_calc_inputs.parameters.get_dict()

        num_bands = w90_params["num_bands"]
        exclude_bands = w90_params.get("exclude_bands", [0])
        start_band = max(exclude_bands) + 1
        end_band = start_band + num_bands - 1

        if self.should_run_yambo_commensurate():
            parent_wkchain = self.ctx.wkchain_yambo_commensurate
            yambo_params = parent_wkchain.inputs.yres.yambo.parameters.get_dict()
        else:
            if self.should_run_yambo_convergence():
                parent_wkchain = get_yambo_converged_workchain(
                    self.ctx.wkchain_yambo_conv
                )
                yambo_params = parent_wkchain.inputs.yres.yambo.parameters.get_dict()
            else:
                # Assume the inputs.parent_folder is generated inside a YamboWorkflow
                parent_folder = inputs.parent_folder
                # The creator is a YamboCalculation, caller is a YamboRestart
                parent_wkchain = parent_folder.creator.caller
                # Assume its caller is a YamboWorkflow
                parent_wkchain = parent_wkchain.caller

        # Reuse converged inputs? Better keep the user provided inputs
        # inputs = parent_wkchain.inputs._construct_attribute_dict(True)

        nscf_wkchain = get_yambo_nscf(parent_wkchain)
        gw_kpoints = (
            get_output_explicit_kpoints(  # pylint: disable=unexpected-keyword-arg
                retrieved=nscf_wkchain.outputs.retrieved,
                metadata={"call_link_label": "get_output_explicit_kpoints"},
            )
        )

        qpkrange = kmapper(  # pylint: disable=unexpected-keyword-arg
            dense_mesh=gw_kpoints,
            coarse_mesh=self.ctx.kpoints_w90,
            start_band=orm.Int(start_band),
            end_band=orm.Int(end_band),
            metadata={"call_link_label": "kmapper"},
        )
        qpkrange = qpkrange.get_list()

        # Set QPkrange in GW parameters
        # yambo_params["variables"]["QPkrange"] = [qpkrange, ""]

        # To be set from input
        if not hasattr(inputs, "QP_subset_dict"):
            inputs.QP_subset_dict = orm.Dict(
                dict={
                    "qp_per_subset": 50,
                    "parallel_runs": 4,
                    "explicit": qpkrange,
                }
            )
        else:
            QP_subset_dict = inputs.QP_subset_dict.get_dict()
            QP_subset_dict["explicit"] = qpkrange
            inputs.QP_subset_dict = orm.Dict(dict=QP_subset_dict)

        inputs.scf.pw.structure = self.ctx.current_structure
        inputs.nscf.pw.structure = self.ctx.current_structure

        inputs.yres.yambo.parameters = orm.Dict(dict=yambo_params)

        inputs.parent_folder = parent_wkchain.outputs.remote_folder

        # Use converged output folder
        settings: dict = inputs.yres.yambo.settings.get_dict()
        # TODO is this correct?
        settings.update({"INITIALISE": False, "COPY_SAVE": True, "COPY_DBS": True})
        inputs.yres.yambo.settings = orm.Dict(dict=settings)

        return inputs

    def run_yambo_qp(self) -> ty.Dict:
        """Run the `YamboRestart` for QP."""
        inputs = self.prepare_yambo_qp_inputs()

        inputs.metadata.call_link_label = "yambo_qp"
        inputs = prepare_process_inputs(YamboWorkflow, inputs)
        running = self.submit(YamboWorkflow, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> for yambo_qp")

        return ToContext(wkchain_yambo_qp=running)

    def inspect_yambo_qp(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the `YamboWorkflow` successfully finished."""
        wkchain = self.ctx.wkchain_yambo_qp

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_YAMBO_QP

    def should_run_ypp_qp(self) -> bool:
        """Whether to run ypp_QP."""

        if "ypp_QP" in self.inputs:
            if "parent_folder" in self.inputs.ypp_QP:
                self.ctx.wkchain_yambo_qp = (
                    self.inputs.ypp_QP.outputs.remote_folder.creator.caller.caller
                )
            QP_list = (
                self.ctx.wkchain_yambo_qp.outputs.splitted_QP_calculations.get_list()
            )
            if len(QP_list) > 1:
                return True

        return False

    def prepare_ypp_inputs_qp(self) -> AttributeDict:
        """Prepare inputs for ypp."""
        inputs = AttributeDict(self.exposed_inputs(YppRestart, namespace="ypp_QP"))

        inputs.ypp.QP_calculations = (
            self.ctx.wkchain_yambo_qp.outputs.splitted_QP_calculations
        )
        inputs.parent_folder = self.ctx.wkchain_yambo_qp.called[0].inputs.parent_folder

        return inputs

    def run_ypp_qp(self) -> ty.Dict:
        """Run the ``YppRestart``."""
        inputs = self.prepare_ypp_inputs_qp()

        inputs.metadata.call_link_label = "ypp_QP"
        inputs = prepare_process_inputs(YppRestart, inputs)
        running = self.submit(YppRestart, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_ypp_QP=running)

    def inspect_ypp_qp(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the ``YppRestart`` successfully finished."""
        wkchain = self.ctx.wkchain_ypp_QP

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_YPP

    def should_run_wannier90_pp(self) -> bool:
        """Whether to run wannier."""
        if self.should_run_ypp() and "nnkp_file" not in self.inputs.ypp.ypp:
            return True

        return False

    def prepare_wannier90_pp_inputs(self) -> AttributeDict:
        """Prepare inputs for wannier90_pp, only for generating nnkp file."""
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace="wannier90")
        )["wannier90"]

        inputs.wannier90.structure = self.ctx.current_structure
        inputs.wannier90.bands_kpoints = self.ctx.current_bands_kpoints

        # Use commensurate kmesh
        if self.ctx.kpoints_w90_input != self.ctx.kpoints_w90:
            set_kpoints(
                inputs, self.ctx.kpoints_w90, process_class=Wannier90BaseWorkChain
            )

        # Only for nnkp, no BandsData for shifting windows
        inputs.shift_energy_windows = False

        # Add `postproc_setup`
        if "settings" in inputs.wannier90:
            settings = inputs.wannier90["settings"].get_dict()
        else:
            settings = {}
        settings["postproc_setup"] = True
        inputs.wannier90["settings"] = settings

        return inputs

    def run_wannier90_pp(self) -> ty.Dict:
        """Run the `Wannier90BaseWorkChain` for postproc."""
        inputs = self.prepare_wannier90_pp_inputs()

        inputs.metadata.call_link_label = "wannier90_pp"
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> for postproc")

        return ToContext(wkchain_wannier90_pp=running)

    def inspect_wannier90_pp(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the `Wannier90BaseWorkChain` successfully finished."""
        wkchain = self.ctx.wkchain_wannier90_pp

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90_PP

    def should_run_ypp(self) -> bool:
        """Whether to run ypp."""
        if "ypp" in self.inputs:
            return True

        return False

    def prepare_ypp_inputs(self) -> AttributeDict:
        """Prepare inputs for ypp."""
        inputs = AttributeDict(self.exposed_inputs(YppRestart, namespace="ypp"))

        # if self.should_run_ypp_qp():
        #    ypp_wkchain = self.ctx.wkchain_ypp_QP
        #    # Working if merge is not needed
        #    inputs.ypp.QP_DB = ypp_wkchain.outputs.QP_DB
        #    inputs.parent_folder = ypp_wkchain.outputs.remote_folder

        if self.should_run_yambo_qp():
            yambo_wkchain = self.ctx.wkchain_yambo_qp
            # Working if merge is not needed
            if "merged_QP" in yambo_wkchain.outputs:
                inputs.ypp.QP_DB = yambo_wkchain.outputs.merged_QP
            else:
                inputs.ypp.QP_DB = yambo_wkchain.outputs.QP_DB
            inputs.parent_folder = self.ctx.wkchain_yambo_qp.called[
                0
            ].inputs.parent_folder

        if self.should_run_wannier90_pp():
            inputs.ypp.nnkp_file = self.ctx.wkchain_wannier90_pp.outputs.nnkp_file

        return inputs

    def run_ypp(self) -> ty.Dict:
        """Run the ``YppRestart``."""
        inputs = self.prepare_ypp_inputs()

        inputs.metadata.call_link_label = "ypp"
        inputs = prepare_process_inputs(YppRestart, inputs)
        running = self.submit(YppRestart, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_ypp=running)

    def inspect_ypp(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the ``YppRestart`` successfully finished."""
        wkchain = self.ctx.wkchain_ypp

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_YPP

    def should_run_wannier90(self) -> bool:
        """Whether to run wannier."""
        if "wannier90" in self.inputs:
            return True

        return False

    def prepare_wannier90_inputs(self) -> AttributeDict:
        """Prepare inputs for wannier90."""
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace="wannier90")
        )

        inputs.structure = self.ctx.current_structure
        inputs.bands_kpoints = self.ctx.current_bands_kpoints

        # Use commensurate kmesh
        if self.ctx.kpoints_w90_input != self.ctx.kpoints_w90:
            set_kpoints(
                inputs, self.ctx.kpoints_w90, process_class=Wannier90OptimizeWorkChain
            )

        return inputs

    def run_wannier90(self) -> ty.Dict:
        """Run the `Wannier90BandsWorkChain`."""
        inputs = self.prepare_wannier90_inputs()

        inputs.metadata.call_link_label = "wannier90"
        inputs = prepare_process_inputs(Wannier90OptimizeWorkChain, inputs)
        running = self.submit(Wannier90OptimizeWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_wannier90=running)

    def inspect_wannier90(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the `Wannier90BandsWorkChain` successfully finished."""
        wkchain = self.ctx.wkchain_wannier90

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

    def should_run_gw2wannier90(self) -> bool:
        """Whether to run gw2wannier90."""
        if "gw2wannier90" in self.inputs:
            return True

        return False

    def prepare_gw2wannier90_inputs(self) -> AttributeDict:
        """Prepare inputs for gw2wannier90."""
        inputs = AttributeDict(
            self.exposed_inputs(Gw2wannier90Calculation, namespace="gw2wannier90")
        )

        if self.should_run_wannier90():
            w90_wkchain = self.ctx.wkchain_wannier90
            inputs.nnkp = w90_wkchain.outputs.wannier90_pp.nnkp_file
            inputs.parent_folder = w90_wkchain.outputs.wannier90.remote_folder

        if self.should_run_ypp():
            inputs.unsorted_eig = self.ctx.wkchain_ypp.outputs.unsorted_eig_file

        return inputs

    def run_gw2wannier90(self) -> ty.Dict:
        """Run the ``gw2wannier90``."""
        inputs = self.prepare_gw2wannier90_inputs()

        inputs.metadata.call_link_label = "gw2wannier90"
        inputs = prepare_process_inputs(Gw2wannier90Calculation, inputs)
        running = self.submit(Gw2wannier90Calculation, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(calc_gw2wannier90=running)

    def inspect_gw2wannier90(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the `Gw2wannier90Calculation` successfully finished."""
        calc = self.ctx.calc_gw2wannier90

        if not calc.is_finished_ok:
            self.report(
                f"{calc.process_label} failed with exit status {calc.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_GW2WANNIER90

    def prepare_wannier90_qp_inputs(self) -> AttributeDict:
        """Prepare inputs for gw2wannier90."""
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90BaseWorkChain, namespace="wannier90_qp")
        )

        inputs.wannier90.structure = self.ctx.current_structure
        inputs.wannier90.bands_kpoints = self.ctx.current_bands_kpoints

        if self.ctx.kpoints_w90_input != self.ctx.kpoints_w90:
            set_kpoints(
                inputs, self.ctx.kpoints_w90, process_class=Wannier90BaseWorkChain
            )

        params = inputs.wannier90.parameters.get_dict()
        params["bands_plot"] = True

        if self.should_run_wannier90():
            w90calc = self.ctx.wkchain_wannier90.outputs.wannier90.remote_folder.creator
            w90calc_params = w90calc.inputs.parameters.get_dict()
            fermi_energy = w90calc_params["fermi_energy"]
            params["fermi_energy"] = fermi_energy

            # TODO I should just restart w/o wannierisation
            # I reuse parameters from previous calculation, overwriting the user inputs
            if inputs.shift_energy_windows:
                keys = ("dis_froz_min", "dis_froz_max", "dis_win_min", "dis_win_max")
                for key in keys:
                    if key in w90calc_params:
                        params[key] = w90calc_params[key]
                inputs.shift_energy_windows = False

        if self.inputs.gw2wannier90.sort_mode in [
            Gw2wannier90SortMode.DEFAULT_AND_CHK,
            Gw2wannier90SortMode.NO_SORT,
        ]:
            params["restart"] = "plot"

        inputs.wannier90.parameters = orm.Dict(dict=params)

        if self.should_run_gw2wannier90():
            inputs.wannier90.remote_input_folder = (
                self.ctx.calc_gw2wannier90.outputs.remote_folder
            )

        return inputs

    def run_wannier90_qp(self) -> ty.Dict:
        """Run the `wannier90 qp`."""
        inputs = self.prepare_wannier90_qp_inputs()

        inputs.metadata.call_link_label = "wannier90_qp"
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}> for wannier90_qp")

        return ToContext(wkchain_wannier90_qp=running)

    def inspect_wannier90_qp(  # pylint: disable=inconsistent-return-statements
        self,
    ) -> ty.Union[None, ExitCode]:
        """Verify that the `Wannier90BaseWorkChain` successfully finished."""
        wkchain = self.ctx.wkchain_wannier90_qp

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90_QP

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

        if "wkchain_yambo_commensurate" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.wkchain_yambo_commensurate,
                    YamboWorkflow,
                    namespace="yambo_commensurate",
                )
            )

        if "wkchain_yambo_qp" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.wkchain_yambo_qp,
                    YamboRestart,
                    namespace="yambo_qp",
                )
            )

        if "wkchain_wannier90_pp" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.wkchain_wannier90_pp,
                    Wannier90BaseWorkChain,
                    namespace="wannier90_pp",
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

        if "wkchain_wannier90" in self.ctx:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.wkchain_wannier90,
                    Wannier90OptimizeWorkChain,
                    namespace="wannier90",
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

        if self.should_run_wannier90():
            bands_w90 = self.outputs["wannier90"]["band_structure"]
            self.out("band_structures.wannier90", bands_w90)

        bands_w90qp = self.outputs["wannier90_qp"]["interpolated_bands"]
        self.out("band_structures.wannier90_qp", bands_w90qp)

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
