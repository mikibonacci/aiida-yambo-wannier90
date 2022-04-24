"""Calculations for gw2wannier90.py."""
import pathlib

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob

from aiida_wannier90.calculations import Wannier90Calculation

from aiida_wannier90_workflows.utils.str import removesuffix

from aiida_yambo_wannier90.common.types import Gw2wannier90SortMode


class Gw2wannier90Calculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the ``gw2wannier90.py``.
    """

    _DEFAULT_INPUT_FOLDER = "unsorted"
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
            required=False,
            help="Remote folder containing amn/mmn/eig/... files.",
        )
        spec.input(
            "unsorted_eig",
            valid_type=orm.SinglefileData,
            required=False,
            help="The seedname.gw.unsorted.eig file.",
        )
        spec.input(
            "nnkp",
            valid_type=orm.SinglefileData,
            required=False,
            help="The seedname.nnkp file.",
        )
        spec.input(
            "sort_mode",
            valid_type=orm.Int,
            serializer=orm.to_aiida_type,
            default=lambda: orm.Int(Gw2wannier90SortMode.NO_SORT),
            help="Modes to sort amn/mmn/eig/chk/... files.",
        )
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Output parameters.",
        )
        spec.output(
            "sort_index",
            valid_type=orm.ArrayData,
            help="Sort index. Note even if sort_mode=NO_SORT, the sort_index is also parsed.",
        )

        spec.exit_code(
            300,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Calculation did not produce all expected output files.",
        )
        spec.exit_code(
            301,
            "ERROR_NO_RETRIEVED_TEMPORARY_FOLDER",
            message="The retrieved temporary folder could not be accessed.",
        )

    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        codeinfo = datastructures.CodeInfo()

        # The input amn/mmn/eig are in `unsorted/` folder,
        # The output amn/mmn/eig are in the current workdir, so the current
        # RemoteData can be directly used by `Wannier90Calculation`.
        w90_default_seedname = removesuffix(
            Wannier90Calculation._DEFAULT_INPUT_FILE,  # pylint: disable=protected-access
            Wannier90Calculation._REQUIRED_INPUT_SUFFIX,  # pylint: disable=protected-access
        )  # actually = aiida
        cmdline_params = [
            "--output_seedname",
            self._DEFAULT_OUTPUT_SEEDNAME,
            f"{self._DEFAULT_INPUT_FOLDER}/{w90_default_seedname}",
        ]
        if self.inputs.sort_mode == Gw2wannier90SortMode.NO_SORT:
            cmdline_params.insert(0, "--no_sort")
        codeinfo.cmdline_params = cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename
        # codeinfo.withmpi = self.inputs.metadata.options.withmpi
        codeinfo.withmpi = False

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]

        local_copy_list = []
        # nnkp file
        nnkp = self.inputs.nnkp
        local_copy_list.append(
            (
                nnkp.uuid,
                nnkp.filename,
                f"{self._DEFAULT_INPUT_FOLDER}/{w90_default_seedname}.nnkp",
            )
        )
        # unsorted.eig file
        unsorted_eig = self.inputs.unsorted_eig
        local_copy_list.append(
            (
                unsorted_eig.uuid,
                unsorted_eig.filename,
                f"{self._DEFAULT_INPUT_FOLDER}/{w90_default_seedname}.gw.unsorted.eig",
            )
        )
        calcinfo.local_copy_list = local_copy_list

        # Files to be sorted by gw2wannier90
        if self.inputs.sort_mode == Gw2wannier90SortMode.DEFAULT:
            extensions = ["eig", "amn", "mmn", "spn"]
        elif self.inputs.sort_mode == Gw2wannier90SortMode.DEFAULT_AND_CHK:
            extensions = ["eig", "amn", "mmn", "spn", "chk"]
        elif self.inputs.sort_mode == Gw2wannier90SortMode.NO_SORT:
            # Only symlink eig to unsorted/aiida.eig
            extensions = ["eig"]

        # symlink the input seedname.[amn|mmn|...] from a remote_folder
        # of Wannier90Calculation to unsorted/aiida.[amn|mmn|...]
        remote_path = pathlib.Path(self.inputs.parent_folder.get_remote_path())
        remote_symlink_list = []
        existed_files = self.inputs.parent_folder.listdir()
        for ext in extensions:
            filename = f"{w90_default_seedname}.{ext}"

            if filename not in existed_files:
                continue

            remote_symlink_list.append(
                (
                    self.inputs.parent_folder.computer.uuid,
                    str(remote_path / filename),
                    f"{self._DEFAULT_INPUT_FOLDER}/{filename}",
                )
            )
        if self.inputs.sort_mode == Gw2wannier90SortMode.NO_SORT:
            # These files are not changed, symlink to workdir (for w90 restart)
            extensions = ["amn", "mmn", "spn", "chk"]
            for ext in extensions:
                filename = f"{w90_default_seedname}.{ext}"

                if filename not in existed_files:
                    continue

                remote_symlink_list.append(
                    (
                        self.inputs.parent_folder.computer.uuid,
                        str(remote_path / filename),
                        filename,
                    )
                )
        calcinfo.remote_symlink_list = remote_symlink_list

        calcinfo.retrieve_temporary_list = [
            f"{self._DEFAULT_OUTPUT_SEEDNAME}.gw2wannier90.raw"
        ]

        calcinfo.retrieve_list = [
            self.metadata.options.output_filename,
        ]

        return calcinfo
