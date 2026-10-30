"""The band path: from the user to the inputs of the workchain, and through the steps."""
from types import SimpleNamespace

import numpy as np
import pytest

from aiida import orm

PATH = [("G", "X"), ("X", "U"), ("K", "G")]
COORDS = {"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.5], "U": [0.625, 0.25, 0.625], "K": [0.375, 0.375, 0.75]}


@pytest.fixture
def builder(generate_builder_inputs):
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    return YamboWannier90WorkChain.get_builder_from_protocol(**generate_builder_inputs("Si"))


def _explicit_list(structure, labels=True):
    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(structure)
    points = [[0, 0, 0], [0.1, 0, 0.1], [0.2, 0, 0.2], [0.5, 0, 0.5]]
    kpoints.set_kpoints(points, labels=[(0, "G"), (3, "X")] if labels else None)
    return kpoints


def test_kpoints_from_path_keeps_segments_and_breaks(generate_structure):
    from aiida_wannier90_workflows.utils.kpoints import get_path_from_kpoints

    from aiida_yambo_wannier90.calculations.functions.kpath import kpoints_from_path

    kpoints = kpoints_from_path(generate_structure("Si"), PATH, COORDS)
    result = get_path_from_kpoints(kpoints).get_dict()

    assert [tuple(segment) for segment in result["path"]] == PATH
    assert {label: np.allclose(coords, COORDS[label]) for label, coords in result["point_coords"].items()} == {
        label: True for label in COORDS
    }


def test_kpoints_from_path_checks_the_labels(generate_structure):
    from aiida_yambo_wannier90.calculations.functions.kpath import kpoints_from_path

    with pytest.raises(ValueError, match="no coordinates for"):
        kpoints_from_path(generate_structure("Si"), [("G", "Z")], COORDS)
    with pytest.raises(ValueError, match="at least one segment"):
        kpoints_from_path(generate_structure("Si"), [], COORDS)


@pytest.mark.parametrize(
    "kwargs, mode, port",
    [
        ({}, "seekpath", None),
        ({"distance": 0.02}, "distance", "bands_kpoints_distance"),
        ({"path": PATH, "point_coords": COORDS}, "path", "kpoint_path"),
        ({"explicit": True}, "explicit_kpoints", "bands_kpoints"),
    ],
)
def test_set_bands_kpath_sets_only_one_source(builder, kwargs, mode, port):
    from aiida_yambo_wannier90.utils.setters import KPATH_INPUTS, set_bands_kpath

    builder.bands_kpoints_distance = orm.Float(0.5)  # something to be replaced
    if kwargs.pop("explicit", False):
        kwargs["explicit_kpoints"] = _explicit_list(builder.structure)

    assert set_bands_kpath(builder, **kwargs) == mode

    assert [name for name in KPATH_INPUTS if name in builder] == ([port] if port else [])


def test_set_bands_kpath_rejects_wrong_combinations(builder):
    from aiida_yambo_wannier90.utils.setters import set_bands_kpath

    with pytest.raises(ValueError, match="together"):
        set_bands_kpath(builder, path=PATH)
    with pytest.raises(ValueError, match="only one"):
        set_bands_kpath(builder, distance=0.1, explicit_kpoints=_explicit_list(builder.structure))
    with pytest.raises(ValueError, match="labels"):
        set_bands_kpath(builder, explicit_kpoints=_explicit_list(builder.structure, labels=False))


def test_the_spec_accepts_one_source_and_rejects_two(builder):
    from aiida_yambo_wannier90.workflows import validate_inputs

    inputs = {"kpoint_path": _explicit_list(builder.structure)}
    assert validate_kpath(inputs) is None
    inputs["bands_kpoints_distance"] = orm.Float(0.1)
    assert "only one" in validate_kpath(inputs)
    assert "labels" in validate_kpath({"bands_kpoints": _explicit_list(builder.structure, labels=False)})
    assert validate_inputs  # imported to check it is public


def validate_kpath(inputs):
    from aiida_yambo_wannier90.workflows import validate_kpath_inputs

    return validate_kpath_inputs(inputs)


@pytest.fixture
def workchain_with(generate_builder_inputs):
    """Instantiate the workchain with a chosen band path."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage import get_manager

    from aiida_yambo_wannier90.utils.setters import set_bands_kpath
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    processes = []

    def _make(**kwargs):
        builder = YamboWannier90WorkChain.get_builder_from_protocol(**generate_builder_inputs("Si"))
        if kwargs.pop("explicit", False):
            kwargs["explicit_kpoints"] = _explicit_list(builder.structure)
        set_bands_kpath(builder, **kwargs)
        process = instantiate_process(get_manager().get_runner(), YamboWannier90WorkChain, **builder._inputs(prune=True))
        processes.append(process)
        return process

    yield _make
    for process in processes:
        process.close()


def test_setup_derives_the_path_from_the_inputs(workchain_with):
    process = workchain_with(path=PATH, point_coords=COORDS)
    process.setup()

    assert process.should_run_seekpath() is False
    assert [tuple(segment) for segment in process.ctx.current_kpoint_path["path"]] == PATH
    assert process.ctx.current_kpoint_list is process.inputs.kpoint_path
    assert process.ctx.current_bands_kpoints is None


def test_an_explicit_list_is_kept_and_not_turned_into_a_path(workchain_with):
    process = workchain_with(explicit=True)
    process.setup()

    assert process.should_run_seekpath() is False
    assert process.ctx.current_bands_kpoints is process.inputs.bands_kpoints
    assert process.ctx.current_kpoint_path is None

    for_dft = SimpleNamespace()
    process._set_bands_kpoints(for_dft)  # pylint: disable=protected-access
    for_qp = SimpleNamespace()
    process._set_bands_kpoints(for_qp, optimize=False)  # pylint: disable=protected-access
    for inputs in (for_dft, for_qp):
        assert inputs.bands_kpoints is process.inputs.bands_kpoints
        assert not hasattr(inputs, "kpoint_path")


def test_the_path_is_given_in_the_form_each_step_wants(workchain_with):
    process = workchain_with(path=PATH, point_coords=COORDS)
    process.setup()

    for_dft = SimpleNamespace()
    process._set_bands_kpoints(for_dft)  # pylint: disable=protected-access
    for_qp = SimpleNamespace()
    process._set_bands_kpoints(for_qp, optimize=False)  # pylint: disable=protected-access

    assert isinstance(for_dft.kpoint_path, orm.KpointsData)  # Wannier90OptimizeWorkChain
    assert isinstance(for_qp.kpoint_path, orm.Dict)  # Wannier90BaseWorkChain
    assert not hasattr(for_dft, "bands_kpoints")


def test_without_a_path_seekpath_runs(workchain_with):
    process = workchain_with()

    assert process.should_run_seekpath() is True
