"""Fixtures for testing workflows."""
import io
import pathlib

import pytest

# pylint: disable=redefined-outer-name,too-many-statements

pytest_plugins = ["aiida.manage.tests.pytest_fixtures"]  # pylint: disable=invalid-name


@pytest.fixture(scope="session")
def filepath_tests():
    """Return the absolute filepath of the `tests` folder.

    .. warning:: if this file moves with respect to the `tests` folder, the implementation should change.

    :return: absolute filepath of `tests` folder which is the basepath for all test resources.
    """
    return pathlib.Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def filepath_fixtures(filepath_tests):
    """Return the absolute filepath to the directory containing the file `fixtures`."""
    return filepath_tests / "fixtures"


@pytest.fixture(scope="function")
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder

    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return a ``Code`` instance configured to run calculations of given entry point on localhost ``Computer``."""

    def _fixture_code(entry_point_name):
        from aiida.common import exceptions
        from aiida.orm import Code

        label = f"test.{entry_point_name}"

        try:
            return Code.objects.get(label=label)  # pylint: disable=no-member
        except exceptions.NotExistent:
            return Code(
                label=label,
                input_plugin_name=entry_point_name,
                remote_computer_exec=[fixture_localhost, "/bin/true"],
            )

    return _fixture_code


@pytest.fixture(scope="session")
def generate_upf_data():
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""

    def _generate_upf_data(element):  # pylint: disable=too-many-locals
        """Return `UpfData` node."""
        from aiida_pseudo.data.pseudo import PseudoPotentialData, UpfData
        import yaml

        # yaml_file = filepath_fixtures / 'pseudos' / 'SSSP_1.1_PBE_efficiency.yaml'
        import aiida_wannier90_workflows

        fixtures_dir = (
            pathlib.Path(aiida_wannier90_workflows.__path__[0])
            / ".."
            / "tests"
            / "fixtures"
        )
        yaml_file = fixtures_dir / "pseudos" / "SSSP_1.1_PBE_efficiency.yaml"

        with open(yaml_file) as file:
            upf_metadata = yaml.load(file, Loader=yaml.FullLoader)

        if element not in upf_metadata:
            raise ValueError(f"Element {element} not found in {yaml_file}")

        filename = upf_metadata[element]["filename"]
        md5 = upf_metadata[element]["md5"]
        z_valence = upf_metadata[element]["z_valence"]
        number_of_wfc = upf_metadata[element]["number_of_wfc"]
        has_so = upf_metadata[element]["has_so"]
        pswfc = upf_metadata[element]["pswfc"]
        ppchi = ""
        for i, l in enumerate(pswfc):  # pylint: disable=invalid-name
            ppchi += f'<PP_CHI.{i+1} l="{l}"/>\n'

        content = (
            '<UPF version="2.0.1">\n'
            "<PP_HEADER\n"
            f'element="{element}"\n'
            f'z_valence="{z_valence}"\n'
            f'has_so="{has_so}"\n'
            f'number_of_wfc="{number_of_wfc}"\n'
            "/>\n"
            "<PP_PSWFC>\n"
            f"{ppchi}"
            "</PP_PSWFC>\n"
            "</UPF>\n"
        )
        stream = io.BytesIO(content.encode("utf-8"))
        upf = UpfData(stream, filename=f"{filename}")

        # I need to hack the md5
        # upf.md5 = md5
        upf.set_attribute(upf._key_md5, md5)  # pylint: disable=protected-access
        # UpfData.store will check md5
        # `PseudoPotentialData` is the parent class of `UpfData`, this will skip md5 check
        super(PseudoPotentialData, upf).store()

        return upf

    return _generate_upf_data


@pytest.fixture(scope="session", autouse=True)
def pseudo_dojo(aiida_profile, generate_upf_data):
    """Create an SSSP pseudo potential family from scratch."""
    from aiida.common.constants import elements
    from aiida.plugins import GroupFactory

    aiida_profile.reset_db()

    # SsspFamily = GroupFactory('pseudo.family.sssp')
    PseudoDojoFamily = GroupFactory("pseudo.family.pseudo_dojo")

    stringency = "standard"
    # Generate SSSP or PseudoDojo
    # label = 'SSSP/1.1/PBE/efficiency'
    # family = SsspFamily(label=label)
    label = "PseudoDojo/0.4/PBE/SR/standard/upf"
    family = PseudoDojoFamily(label=label)
    family.store()

    cutoffs = {}
    upfs = []

    for values in elements.values():
        element = values["symbol"]
        if element in ["X"]:
            continue
        try:
            upf = generate_upf_data(element)
        except ValueError:
            continue

        upfs.append(upf)

        cutoffs[element] = {
            "cutoff_wfc": 30.0,
            "cutoff_rho": 240.0,
        }

    family.add_nodes(upfs)
    family.set_cutoffs(cutoffs, stringency, unit="Ry")

    return family


@pytest.fixture
def serialize_builder():
    """Serialize the given process builder into a dictionary with nodes turned into their value representation.

    :param builder: the process builder to serialize
    :return: dictionary
    """
    from aiida_wannier90_workflows.utils.workflows.builder import serializer

    def _serializer(node):
        return serializer(node, show_pk=False)

    return _serializer


@pytest.fixture
def generate_structure():
    """Return a ``StructureData`` representing either bulk silicon or a water molecule."""

    def _generate_structure(structure_id="Si"):
        """Return a ``StructureData`` representing bulk silicon or a snapshot of a single water molecule dynamics.

        :param structure_id: identifies the ``StructureData`` you want to generate. Either 'Si' or 'H2O' or 'GaAs'.
        """
        from aiida.orm import StructureData

        if structure_id == "Si":
            param = 5.43
            cell = [
                [param / 2.0, param / 2.0, 0],
                [param / 2.0, 0, param / 2.0],
                [0, param / 2.0, param / 2.0],
            ]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
            structure.append_atom(
                position=(param / 4.0, param / 4.0, param / 4.0),
                symbols="Si",
                name="Si",
            )
        elif structure_id == "H2O":
            structure = StructureData(
                cell=[
                    [5.29177209, 0.0, 0.0],
                    [0.0, 5.29177209, 0.0],
                    [0.0, 0.0, 5.29177209],
                ]
            )
            structure.append_atom(
                position=[12.73464656, 16.7741411, 24.35076238], symbols="H", name="H"
            )
            structure.append_atom(
                position=[-29.3865565, 9.51707929, -4.02515904], symbols="H", name="H"
            )
            structure.append_atom(
                position=[1.04074437, -1.64320127, -1.27035021], symbols="O", name="O"
            )
        elif structure_id == "GaAs":
            structure = StructureData(
                cell=[
                    [0.0, 2.8400940897, 2.8400940897],
                    [2.8400940897, 0.0, 2.8400940897],
                    [2.8400940897, 2.8400940897, 0.0],
                ]
            )
            structure.append_atom(position=[0.0, 0.0, 0.0], symbols="Ga", name="Ga")
            structure.append_atom(
                position=[1.42004704485, 1.42004704485, 4.26014113455],
                symbols="As",
                name="As",
            )
        elif structure_id == "BaTiO3":
            structure = StructureData(
                cell=[
                    [3.93848606, 0.0, 0.0],
                    [0.0, 3.93848606, 0.0],
                    [0.0, 0.0, 3.93848606],
                ]
            )
            structure.append_atom(position=[0.0, 0.0, 0.0], symbols="Ba", name="Ba")
            structure.append_atom(
                position=[1.969243028987539, 1.969243028987539, 1.969243028987539],
                symbols="Ti",
                name="Ti",
            )
            structure.append_atom(
                position=[0.0, 1.969243028987539, 1.969243028987539],
                symbols="O",
                name="O",
            )
            structure.append_atom(
                position=[1.969243028987539, 1.969243028987539, 0.0],
                symbols="O",
                name="O",
            )
            structure.append_atom(
                position=[1.969243028987539, 0.0, 1.969243028987539],
                symbols="O",
                name="O",
            )
        else:
            raise KeyError(f"Unknown structure_id='{structure_id}'")
        return structure

    return _generate_structure


@pytest.fixture
def generate_builder_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``Wannier90BandsWorkChain.get_builder_from_protocol()`` method."""

    def _generate_builder_inputs(structure_id="Si"):
        inputs = {
            "codes": {
                "pw": fixture_code("quantumespresso.pw"),
                "pw2wannier90": fixture_code("quantumespresso.pw2wannier90"),
                "wannier90": fixture_code("wannier90.wannier90"),
                "projwfc": fixture_code("quantumespresso.projwfc"),
                "opengrid": fixture_code("quantumespresso.opengrid"),
                "yambo": fixture_code("yambo.yambo"),
                "p2y": fixture_code("yambo.p2y"),
                "ypp": fixture_code("yambo.ypp"),
                "gw2wannier90": fixture_code("yambo_wannier90.gw2wannier90"),
            },
            "structure": generate_structure(structure_id=structure_id),
        }
        return inputs

    return _generate_builder_inputs
