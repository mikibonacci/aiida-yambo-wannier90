"""
Parsers provided by aiida_skeaf.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
import pathlib
import re
import typing as ty

import numpy as np

from aiida import orm
from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.parsers.parser import Parser

from aiida_wannier90_workflows.utils.str import removeprefix, removesuffix

from aiida_yambo_wannier90.calculations.gw2wannier90 import Gw2wannier90Calculation


class Gw2wannier90Parser(Parser):
    """
    Parser class for parsing output of ``gw2wannier90.py``.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a Gw2wannier90Calculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, Gw2wannier90Calculation):
            raise exceptions.ParsingError("Can only parse Gw2wannier90Calculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        output_filename = self.node.get_option("output_filename")

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = [
            Gw2wannier90Calculation._DEFAULT_OUTPUT_FILE,  # pylint: disable=protected-access
        ]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # parse `gw2wannier90.out`
        self.logger.info(f"Parsing '{output_filename}'")
        with self.retrieved.open(output_filename, "r") as handle:
            output_node = parse_gw2wannier90_out(handle.readlines())
        self.out("output_parameters", output_node)

        # attach sort_index
        self.logger.info("Attaching sort_index")
        self.attach_sort_index(**kwargs)

        return ExitCode(0)

    def attach_sort_index(  # pylint: disable=inconsistent-return-statements
        self, **kwargs
    ):
        """Attach RemoteData for extracted bxsf."""

        retrieve_temporary_list = self.node.get_attribute(
            "retrieve_temporary_list", None
        )

        # If temporary files were specified, check that we have them
        if retrieve_temporary_list:
            try:
                retrieved_temporary_folder = kwargs["retrieved_temporary_folder"]
            except KeyError:
                return self.exit(self.exit_codes.ERROR_NO_RETRIEVED_TEMPORARY_FOLDER)

        filename = f"{self.node.process_class._DEFAULT_OUTPUT_SEEDNAME}.gw2wannier90.raw"  # pylint: disable=protected-access
        if filename not in retrieve_temporary_list:
            return self.exit(self.exit_codes.ERROR_MISSING_OUTPUT_FILES)

        with open(pathlib.Path(retrieved_temporary_folder) / filename) as handle:
            filecontent = handle.readlines()
            sort_array = parse_gw2wannier90_raw(filecontent)
            self.out("sort_index", sort_array)


def parse_gw2wannier90_out(filecontent: ty.List[str]) -> orm.Dict:
    """Parse `gw2wannier90.out`."""
    parameters = {}

    regexs = {
        "timestamp_started": re.compile(r"Started on\s*(.+)"),
        "num_kpoints": re.compile(r"Kpoints number:\s*([0-9]+)"),
    }

    for line in filecontent:
        for key, reg in regexs.items():
            match = reg.match(line.strip())
            if match:
                parameters[key] = match.group(1)
                regexs.pop(key, None)
                break

    parameters["num_kpoints"] = int(parameters["num_kpoints"])

    return orm.Dict(dict=parameters)


def parse_gw2wannier90_raw(filecontent: ty.List[str]) -> orm.ArrayData:
    """Parse `aiida.gw2wannier90.raw`."""
    sort_index = []
    read_index = False

    for line in filecontent:
        if "Writing sorting list" in line:
            read_index = True
            continue
        if "------------------------------" in line and read_index:
            read_index = False
        if read_index:
            line = line.strip()
            if line.startswith("["):
                row = removeprefix(line, "[")
                if line.endswith("]"):
                    row = removesuffix(row, "]")
                    row = row.split()
                    sort_index.append([int(_) for _ in row])
                    continue
            elif line.endswith("]"):
                row += " " + removesuffix(line, "]")
                row = row.split()
                sort_index.append([int(_) for _ in row])
            else:
                row += " " + line

    sort_array = orm.ArrayData()
    sort_array.set_array("sort_index", np.array(sort_index))

    return sort_array
