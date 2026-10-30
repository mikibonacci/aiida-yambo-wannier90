#!/usr/bin/env python
"""Base class for Yambo+Wannier90 workflow."""
import copy
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
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90BaseWorkChain,
    Wannier90OptimizeWorkChain,
)

from aiida_yambo.workflows.yamboconvergence import YamboConvergence
from aiida_yambo.workflows.yambowf import YamboWorkflow
from aiida_yambo.workflows.ypprestart import YppRestart

from aiida_yambo_wannier90.calculations.functions.kpath import kpoint_path_dict
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

from .validation import KPATH_INPUTS, validate_inputs, validate_kpath_inputs  # noqa: F401  # pylint: disable=unused-import

__all__ = ["validate_inputs", "validate_kpath_inputs", "YamboWannier90WorkChain"]


# pylint: disable=too-many-lines
# pylint: disable=fixme
# TODO remove this todo disable


def _copy_namespaces(inputs: dict) -> dict:
    """Copy a nested dictionary of inputs: the dictionaries are new, the (AiiDA) values are the same objects."""
    return {
        key: _copy_namespaces(value) if isinstance(value, dict) else value
        for key, value in inputs.items()
    }


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
            "kpoint_path",
            valid_type=orm.KpointsData,
            required=False,
            help=(
                "High-symmetry points of the band path, with `labels`: wannier90 interpolates the bands along the "
                "segments between consecutive labelled points. Specify only one of `kpoint_path`, `bands_kpoints` "
                "and `bands_kpoints_distance`; with none of them the workchain runs seekpath."
            ),
        )
        spec.input(
            "bands_kpoints",
            valid_type=orm.KpointsData,
            required=False,
            help=(
                "Explicit list of k-points (with `labels`) where the bands are computed, exactly these points "
                "(for example to compare point by point with the Quantum ESPRESSO bands). "
                "Specify only one of `kpoint_path`, `bands_kpoints` and `bands_kpoints_distance`."
            ),
        )
        spec.input(
            "bands_kpoints_distance",
            valid_type=orm.Float,
            serializer=orm.to_aiida_type,
            required=False,
            help=(
                "Minimum kpoints distance for seekpath to generate the path (in the primitive cell). "
                "Specify only one of `kpoint_path`, `bands_kpoints` and `bands_kpoints_distance`."
            ),
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
                    "Inputs for the `YamboWorkflow` of the yambo QP calculation (`scf`, `nscf`, `qp`, `QP_subsets_dict`...). "
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
        from importlib.resources import files

        from . import protocols

        return files(protocols) / "yambo_wannier90.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=too-many-statements,too-many-locals,too-many-arguments
        cls,
        codes: ty.Mapping[str, ty.Union[orm.Code, str, int]],
        structure: orm.StructureData,
        *,
        protocol: str = None,
        overrides: dict = None,
        pseudo_family: str = "PseudoDojo/0.4/PBE/SR/standard/upf",
        exclude_semicore: bool = False,
        electronic_type=ElectronicType.METAL,
        spin_type=SpinType.NONE,
        wannier_projection_type: WannierProjectionType = WannierProjectionType.ATOMIC_PROJECTORS_QE,
        disentanglement_type: WannierDisentanglementType = None,
        frozen_type: WannierFrozenType = None,
        NLCC: bool = True,
        RIM_v: bool = True,
        RIM_W: bool = False,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param codes: the codes, by label, pk or node: ``pw``, ``pw2wannier90``, ``wannier90``, ``yambo``, ``p2y``, ``ypp``,
            ``gw2wannier90`` and, for the projectors of Quantum ESPRESSO, ``projwfc``. The dictionary is not modified.
        :param structure: the structure.
        :param protocol: ``fast``, ``moderate`` (default) or ``precise``: it is used for the yambo convergence and the
            QP calculation and for the wannier90 workflows.
        :param overrides: nested dictionary, one key for each stage: ``yambo`` (the ``YamboConvergence``: e.g.
            ``{"ywfl": {"qp": {...}}}``), ``yambo_qp``, ``ypp``, ``wannier90`` (the ``Wannier90OptimizeWorkChain``) and
            ``wannier90_qp``, and the keys of this workchain. The values are merged in the inputs of the stage.
        :param wannier_projection_type, disentanglement_type, frozen_type: how the Wannier functions are built; the
            last two are guessed from the first one if not given. They are used by both wannier90 runs.
        :return: a builder; see :mod:`aiida_yambo_wannier90.utils.setters` to change it, and ``print_summary``.
        """
        # pylint: disable=import-outside-toplevel,protected-access
        from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
        from aiida_wannier90_workflows.utils.workflows.builder.projections import (
            guess_wannier_projection_types,
        )
        from aiida_wannier90_workflows.utils.workflows.builder.submit import (
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
        missing = [name for name in required_codes if name not in codes]
        if missing:
            raise ValueError(f"`codes` must contain {required_codes}, missing {missing}")
        # not the caller's dictionary
        codes = {
            key: code if isinstance(code, orm.Code) else orm.load_code(code)
            for key, code in codes.items()
        }

        type_check(structure, orm.StructureData)

        protocol = protocol or cls.get_default_protocol()
        overrides = copy.deepcopy(overrides) if overrides else {}
        stage_overrides = {
            stage: overrides.pop(stage, {})
            for stage in ("yambo", "yambo_qp", "ypp", "wannier90", "wannier90_qp")
        }
        inputs = cls.get_protocol_inputs(protocol, overrides)
        inputs["structure"] = structure

        # The types of Wannier functions, the same for the two wannier90 runs
        _, disentanglement_type, frozen_type = guess_wannier_projection_types(
            electronic_type, wannier_projection_type, disentanglement_type, frozen_type
        )

        # Prepare yambo
        yambo_overrides = recursive_merge(
            {
                "ywfl": {
                    "scf": {"pseudo_family": pseudo_family},
                    "nscf": {"pseudo_family": pseudo_family},
                },
            },
            stage_overrides["yambo"],
        )
        yambo_builder = YamboConvergence.get_builder_from_protocol(
            pw_code=codes["pw"],
            preprocessing_code=codes["p2y"],
            code=codes["yambo"],
            protocol=protocol,
            protocol_qe=protocol,
            structure=structure,
            electronic_type=electronic_type,
            spin_type=spin_type,
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
        wannier_builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            codes,
            structure,
            protocol=protocol,
            overrides=stage_overrides["wannier90"] or None,
            pseudo_family=pseudo_family,
            electronic_type=electronic_type,
            spin_type=spin_type,
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

        # The yambo QP calculation: a YamboWorkflow, as the one of the convergence (a copy of the namespaces; the nodes
        # are shared). The QP to compute are chosen at runtime (see `QP_subsets_dict` and `set_qp_subsets`).
        inputs["yambo_qp"] = _copy_namespaces(inputs["yambo"]["ywfl"])
        inputs["yambo_qp"].pop("clean_workdir", None)

        # Ypp; without a parent_folder for now. We should set it during the input preparation
        ypp_builder = YppRestart.get_builder_from_protocol(
            code=codes["ypp"],
            protocol="Wannier",
        )
        inputs["ypp"] = ypp_builder._inputs(prune=True)
        inputs["ypp"].pop("clean_workdir", None)

        # Prepare gw2wannier90
        inputs["gw2wannier90"] = {
            "code": codes["gw2wannier90"],
        }

        # Prepare wannier90_qp: the same Wannier functions as the first run, then the settings which depend on the
        # first run are copied (`sync_wannier90_qp`)
        qp_overrides = recursive_merge(
            {"meta_parameters": {"exclude_semicore": exclude_semicore}},
            stage_overrides["wannier90_qp"],
        )
        wannier90_qp_builder = Wannier90BaseWorkChain.get_builder_from_protocol(
            code=codes["wannier90"],
            structure=structure,
            protocol=protocol,
            overrides=qp_overrides,
            pseudo_family=pseudo_family,
            electronic_type=electronic_type,
            spin_type=spin_type,
            projection_type=wannier_projection_type,
            disentanglement_type=disentanglement_type,
            frozen_type=frozen_type,
        )
        params = wannier90_qp_builder.wannier90.parameters.get_dict()
        params["bands_plot"] = True
        wannier90_qp_builder.wannier90.parameters = orm.Dict(params)
        inputs["wannier90_qp"] = wannier90_qp_builder._inputs(prune=True)
        inputs["wannier90_qp"]["wannier90"].pop("structure", None)
        inputs["wannier90_qp"].pop("clean_workdir", None)

        builder = cls.get_builder()
        builder = recursive_merge_builder(builder, inputs)

        # the overrides of the stages which have no builder of their own
        user = {
            stage: stage_overrides[stage] for stage in ("yambo_qp", "ypp") if stage_overrides[stage]
        }
        if user:
            builder = recursive_merge_builder(builder, user)

        from aiida_yambo_wannier90.utils.setters import sync_wannier90_qp

        sync_wannier90_qp(builder)
        return builder

    def setup(self) -> None:  # pylint: disable=inconsistent-return-statements
        """Initialize context variables."""

        self.ctx.current_structure = self.inputs.structure

        # The band path, as the steps need it. `current_kpoint_path` is the `Dict` of wannier90 (`path` and
        # `point_coords`), `current_kpoint_list` the labelled `KpointsData` of the high-symmetry points (for the
        # `Wannier90OptimizeWorkChain`), `current_bands_kpoints` an explicit list of points; seekpath fills the first two.
        self.ctx.current_kpoint_path = None
        self.ctx.current_kpoint_list = None
        self.ctx.current_bands_kpoints = None
        if "kpoint_path" in self.inputs:
            self.ctx.current_kpoint_list = self.inputs.kpoint_path
            self.ctx.current_kpoint_path = kpoint_path_dict(
                self.inputs.kpoint_path, metadata={"call_link_label": "kpoint_path_dict"}
            )
        elif "bands_kpoints" in self.inputs:
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
        """Run seekpath if neither `kpoint_path` nor `bands_kpoints` are provided."""
        return "kpoint_path" not in self.inputs and "bands_kpoints" not in self.inputs

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
        
        self.ctx.current_kpoint_list = result["explicit_kpoints"]

        # Add `kpoint_path` for Wannier bands
        self.ctx.current_kpoint_path = kpoint_path_dict(
            result["explicit_kpoints"], metadata={"call_link_label": "kpoint_path_dict"}
        )

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

        if "yambo_qp" in self.inputs.keys():
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
        if "QP_subsets_dict" in inputs:
            del inputs.QP_subsets_dict
        inputs.scf.pw.structure = self.ctx.current_structure
        inputs.nscf.pw.structure = self.ctx.current_structure

        if self.should_run_yambo_convergence():
            converged_wkchain = get_yambo_converged_workchain(self.ctx.wkchain_yambo_conv)
            inputs.qp.yambo.parameters = converged_wkchain.inputs.qp.yambo.parameters

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
        yambo_params = inputs.qp.yambo.parameters.get_dict()

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
            yambo_params = parent_wkchain.inputs.qp.yambo.parameters.get_dict()
        else:
            if self.should_run_yambo_convergence():
                parent_wkchain = get_yambo_converged_workchain(
                    self.ctx.wkchain_yambo_conv
                )
                yambo_params = parent_wkchain.inputs.qp.yambo.parameters.get_dict()
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

        # The QPs to compute are the ones of the coarse Wannier90 mesh (`explicit`); the other keys of `QP_subsets_dict`
        # (`qp_per_subset`, `parallel_runs`, `parallelism`, `resources`...) are the user ones, with these defaults.
        # Any other source of QP given by the user is replaced.
        QP_subsets_dict = (
            inputs.QP_subsets_dict.get_dict()
            if "QP_subsets_dict" in inputs
            else {"qp_per_subset": 50, "parallel_runs": 4}
        )
        for source in ("boundaries", "range_QP"):
            QP_subsets_dict.pop(source, None)
        QP_subsets_dict["explicit"] = qpkrange
        inputs.QP_subsets_dict = orm.Dict(QP_subsets_dict)
        inputs.scf.pw.structure = self.ctx.current_structure
        inputs.nscf.pw.structure = self.ctx.current_structure

        inputs.qp.yambo.parameters = orm.Dict(yambo_params)

        inputs.parent_folder = parent_wkchain.outputs.remote_folder

        # Use converged output folder
        settings: dict = inputs.qp.yambo.settings.get_dict() if "settings" in inputs.qp.yambo else {}
        settings.update({"INITIALISE": False, "COPY_SAVE": True, "COPY_DBS": True})
        inputs.qp.yambo.settings = orm.Dict(settings)

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
        
        params = inputs.wannier90.parameters.get_dict()
        params["bands_plot"] = False
        inputs.wannier90.parameters = orm.Dict(params)
        
        # no bands are computed in the postprocessing, but the path is kept in the inputs
        if self.ctx.current_kpoint_path is not None:
            inputs.wannier90.kpoint_path = self.ctx.current_kpoint_path

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

        if self.should_run_yambo_qp():
            yambo_wkchain = self.ctx.wkchain_yambo_qp
            # `merged_QP` if the QP splitter ran, else the `QP_db` of the single GW calculation
            if "merged_QP" in yambo_wkchain.outputs:
                inputs.ypp.QP_DB = yambo_wkchain.outputs.merged_QP
            else:
                inputs.ypp.QP_DB = yambo_wkchain.outputs.QP_db
            inputs.parent_folder = self.ctx.wkchain_yambo_qp.outputs.remote_folder.creator.outputs.remote_folder

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

    def _set_bands_kpoints(self, inputs: AttributeDict, optimize: bool = True) -> None:
        """Set the band path on the inputs of a wannier90 step: the explicit list of points or the high-symmetry path.

        The `Wannier90OptimizeWorkChain` (`optimize=True`) takes the path as a labelled `KpointsData` (`kpoint_path`),
        the `Wannier90BaseWorkChain` as the `Dict` of wannier90; both take an explicit list as `bands_kpoints`.
        """
        if self.ctx.current_bands_kpoints is not None:
            inputs.bands_kpoints = self.ctx.current_bands_kpoints
        elif optimize:
            inputs.kpoint_path = self.ctx.current_kpoint_list
        else:
            inputs.kpoint_path = self.ctx.current_kpoint_path

    def prepare_wannier90_inputs(self) -> AttributeDict:
        """Prepare inputs for wannier90."""
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace="wannier90")
        )

        inputs.structure = self.ctx.current_structure
        self._set_bands_kpoints(inputs)

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
        self._set_bands_kpoints(inputs.wannier90, optimize=False)

        # The mesh of the first wannier90 run (which can have been made commensurate with the GW one): the
        # QP run must use the same, since it reads its matrices. Without the first run, the user's one.
        if self.ctx.kpoints_w90 is not None:
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

        # the context entry of each stage, its class and the namespace of its outputs
        stages = (
            ("wkchain_yambo_conv", YamboConvergence, "yambo"),
            ("wkchain_yambo_commensurate", YamboWorkflow, "yambo_commensurate"),
            ("wkchain_yambo_qp", YamboWorkflow, "yambo_qp"),
            ("wkchain_wannier90_pp", Wannier90BaseWorkChain, "wannier90_pp"),
            ("wkchain_ypp", YppRestart, "ypp"),
            ("wkchain_wannier90", Wannier90OptimizeWorkChain, "wannier90"),
            ("calc_gw2wannier90", Gw2wannier90Calculation, "gw2wannier90"),
        )
        for name, process_class, namespace in stages:
            if name in self.ctx:
                self.out_many(self.exposed_outputs(self.ctx[name], process_class, namespace=namespace))

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
