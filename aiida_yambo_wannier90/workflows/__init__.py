#!/usr/bin/env python
"""Base class for Yambo+Wannier90 workflow."""
import pathlib
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ExitCode, ProcessBuilder, ToContext, WorkChain, if_

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

from aiida_yambo_wannier90.calculations.kmesh import (
    find_commensurate_meshes,
    get_output_explicit_kpoints,
    kmapper,
)

__all__ = ["validate_inputs", "YamboWannier90WorkChain"]


def validate_inputs(
    inputs: AttributeDict, ctx=None  # pylint: disable=unused-argument
) -> None:
    """Validate the inputs of the entire input namespace."""


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
        spec.expose_inputs(
            YamboConvergence,
            namespace="yambo",
            exclude=("clean_workdir", "structure"),
            namespace_options={
                "help": "Inputs for the `YamboConvergence` for yambo calculation."
            },
        )
        spec.expose_inputs(
            Wannier90OptimizeWorkChain,
            namespace="wannier90",
            exclude=("clean_workdir", "structure"),
            namespace_options={
                "help": "Inputs for the `Wannier90OptimizeWorkChain` for wannier90 calculation."
            },
        )

        spec.inputs.validator = validate_inputs

        spec.expose_outputs(YamboConvergence, namespace="yambo")
        spec.expose_outputs(Wannier90OptimizeWorkChain, namespace="wannier")

        spec.outline(
            cls.setup,
            cls.run_yambo_convergence,
            cls.inspect_yambo_convergence,
            cls.setup_kmesh,
            cls.run_wannier,
            cls.inspect_wannier,
            cls.run_yambo_qp,
            cls.inspect_yambo_qp,
            cls.run_ypp,
            cls.inspect_ypp,
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
        bxsf: orm.RemoteData,
        num_electrons: int,
        protocol: str = None,
        overrides: dict = None,
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

        required_codes = ["pw", "pw2wannier90", "wannier90", "yambo", "p2y", "ypp"]
        if not all(_ in codes for _ in required_codes):
            raise ValueError(f"`codes` must contain {required_codes}")

        for key, code in codes.items():
            if not isinstance(code, orm.Code):
                codes[key] = orm.load_code(code)

        type_check(bxsf, orm.RemoteData)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        # TODO yamboworkflow run seekpath -> primitive cell?
        ecutwfc = 80
        pseudo_family = "PseudoDojo/0.4/PBE/SR/standard/upf"
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
            codes["pw"],
            codes["p2y"],
            codes["yambo"],
            protocol="moderate",
            structure=structure,
            overrides=overrides,
            # parent_folder=load_node(225176).outputs.remote_folder,
        )
        #####

        inputs["wan2skeaf"]["code"] = codes["wan2skeaf"]
        inputs["skeaf"]["code"] = codes["skeaf"]
        inputs["bxsf"] = bxsf

        wan2skeaf_parameters = inputs["wan2skeaf"]["parameters"]
        wan2skeaf_parameters["num_electrons"] = num_electrons
        inputs["wan2skeaf"]["parameters"] = orm.Dict(dict=wan2skeaf_parameters)

        builder = cls.get_builder()
        builder = recursive_merge_builder(builder, inputs)

        return builder

    def setup(self) -> None:
        """Initialize context variables."""

        # Commensurate meshes for GW and W90
        self.ctx.kpoints_gw = None
        self.ctx.kpoints_w90 = None

    def run_yambo_convergence(self) -> ty.Dict:
        """Run the `YamboConvergence`."""
        inputs = AttributeDict(self.exposed_inputs(YamboConvergence, namespace="yambo"))

        inputs.metadata.call_link_label = "yambo_convergence"

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

    def setup_kmesh(self) -> None:
        """Find commensurate kmeshes for both Yambo and wannier90."""
        wkchain_gw = self.ctx.wkchain_yambo_conv
        kpoints_gw = wkchain_gw.inputs.nscf.kpoints
        kpoints_w90 = self.inputs.wannier.nscf.kpoints

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

        # Use theses meshes before submitting the corresponding workflow
        self.ctx.kpoints_gw = create_kpoints_from_mesh(kmesh_gw)
        self.ctx.kpoints_w90 = get_explicit_kpoints_from_mesh(kmesh_w90)

    def run_wannier(self) -> ty.Dict:
        """Run the `Wannier90BandsWorkChain`."""
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90OptimizeWorkChain, namespace="wannier")
        )
        inputs.metadata.call_link_label = "wannier"

        # Use commensurate kmesh
        inputs.nscf.kpoints = self.ctx.kpoints_w90
        inputs.wannier90.wannier90.kpoints = self.ctx.kpoints_w90

        inputs = prepare_process_inputs(Wannier90OptimizeWorkChain, inputs)
        running = self.submit(Wannier90OptimizeWorkChain, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_wannier=running)

    def inspect_wannier(self) -> ty.Union[None, ExitCode]:
        """Verify that the `Wannier90BandsWorkChain` successfully finished."""
        wkchain = self.ctx.wkchain_wannier

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER

    def prepare_yambo_qp_inputs(self) -> AttributeDict:
        """Prepare inputs for yambo QP."""
        # TODO the converged input for YamboWorkflow?
        inputs = AttributeDict(
            self.exposed_inputs(YamboConvergence, namespace="yambo")
        ).yres

        # Prepare QPkrange
        w90_params = (
            self.ctx.wkchain_wannier.inputs.wannier90.wannier90.parameters.get_dict()
        )
        num_bands = w90_params["num_bands"]
        exclude_bands = w90_params.get("exclude_bands", [0])
        start_band = max(exclude_bands) + 1
        end_band = start_band + num_bands - 1

        p2y_nscf_wkchain = self.ctx.wkchain_yambo_conv.get_outgoing(
            link_label_filter="p2y",
        ).get_outgoing(
            link_label_filter="nscf",
        )

        gw_kpoints = get_output_explicit_kpoints(p2y_nscf_wkchain.outputs.retrieved)
        qpkrange = kmapper(
            dense_mesh=gw_kpoints,
            coarse_mesh=self.ctx.kpoints_w90,
            start_band=orm.Int(start_band),
            end_band=orm.Int(end_band),
        )

        # TODO set QPkrange in GW parameters

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

    def prepare_ypp_inputs(self) -> AttributeDict:
        """Prepare inputs for ypp."""
        # TODO the converged input for YamboWorkflow?
        inputs = AttributeDict(
            self.exposed_inputs(YamboConvergence, namespace="yambo")
        ).yres

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

    def run_wannier_qp(self) -> ty.Dict:
        """Run the `ypp`."""
        inputs = self.prepare_ypp_inputs()

        inputs.metadata.call_link_label = "wannier_qp"
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, inputs)
        running = self.submit(YppRestart, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(wkchain_wannier_qp=running)

    def inspect_wannier_qp(self) -> ty.Union[None, ExitCode]:
        """Verify that the `Wannier90BaseWorkChain` successfully finished."""
        wkchain = self.ctx.wkchain_wannier_qp

        if not wkchain.is_finished_ok:
            self.report(
                f"{wkchain.process_label} failed with exit status {wkchain.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER_QP

    def results(self) -> None:
        """Attach the relevant output nodes."""

        self.out_many(
            self.exposed_outputs(
                self.ctx.wkchain_yambo_conv,
                YamboConvergence,
                namespace="yambo",
            )
        )

        self.out_many(
            self.exposed_outputs(
                self.ctx.wkchain_wannier,
                Wannier90OptimizeWorkChain,
                namespace="wannier",
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
