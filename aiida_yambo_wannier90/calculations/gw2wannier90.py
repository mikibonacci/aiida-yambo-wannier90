"""Calculations for gw2wannier90.py."""
import pathlib

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob


class Gw2wannier90Calculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the ``gw2wannier90.py``.
    """

    _DEFAULT_INPUT_SEEDNAME = "aiida.unsorted"
    _DEFAULT_OUTPUT_SEEDNAME = "aiida"
    _DEFAULT_OUTPUT_FILE = "gw2wannier90.out"

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"][
            "parser_name"
        ].default = "yambo_wannier90.gw2wannier90"

        # new ports
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILE,
        )
        spec.input(
            "parent_folder",
            valid_type=orm.RemoteData,
            help="Remote folder containing win/amn/mmn/eig/... files.",
        )
        spec.input(
            "unsorted_eig",
            valid_type=orm.SinglefileData,
            help="The seedname.gw.unsorted.eig file.",
        )
        spec.input(
            "nnkp",
            valid_type=orm.SinglefileData,
            help="The seedname.nnkp file.",
        )
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Output parameters.",
        )

        spec.exit_code(
            300,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Calculation did not produce all expected output files.",
        )

    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        codeinfo = datastructures.CodeInfo()

        cmdline_params = [self._DEFAULT_INPUT_SEEDNAME]
        codeinfo.cmdline_params = cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename
        # codeinfo.withmpi = self.inputs.metadata.options.withmpi
        codeinfo.withmpi = False

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        # calcinfo.local_copy_list = [
        #     (
        #         self.inputs.file1.uuid,
        #         self.inputs.file1.filename,
        #         self.inputs.file1.filename,
        #     ),
        # ]

        # symlink the input bxsf
        remote_path = self.inputs.parent_folder.get_remote_path()
        calcinfo.remote_symlink_list = [
            (
                self.inputs.bxsf.computer.uuid,
                remote_path,
                self._DEFAULT_INPUT_BXSF,
            ),
        ]

        calcinfo.retrieve_list = [
            self.metadata.options.output_filename,
        ]

        return calcinfo
