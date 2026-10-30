"""The examples are what users copy: they run with test nodes, and the inputs they build must be valid."""
import pytest

from .example_harness import install_submit, make_environment, run_example


@pytest.fixture
def environment(aiida_profile, aiida_localhost, monkeypatch):
    environment = make_environment(aiida_localhost)
    install_submit(monkeypatch, environment.submitted)
    return environment


RUNS = (
    [("01_wannier90_dft_bands.py", {})]
    + [("02_custom_kpath.py", {"KPATH_MODE": mode}) for mode in ("seekpath", "distance", "path", "kpoints", "explicit")]
    + [
        ("03_gw_wannier90_from_scratch.py", {}),
        ("04_from_existing_gw.py", {"STAGE": "yambo_qp", "PARENT_PK": "{PARENT_PK}"}),
        ("04_from_existing_gw.py", {"STAGE": "yambo_qp", "GW_MESH": [4, 4, 4]}),
        ("04_from_existing_gw.py", {"STAGE": "ypp", "PARENT_PK": "{PARENT_PK}", "QP_DB_PK": "{QP_DB_PK}"}),
        ("04_from_existing_gw.py", {"STAGE": "wannier90_qp", "REMOTE_PK": "{REMOTE_PK}"}),
    ]
)


@pytest.mark.parametrize("script, overrides", RUNS, ids=[f"{s}-{o}" for s, o in RUNS])
def test_example_builds_valid_inputs(environment, script, overrides):
    overrides = {
        key: environment.settings[value.strip("{}")] if isinstance(value, str) and value.startswith("{") else value
        for key, value in overrides.items()
    }

    run_example(script, environment.settings, overrides)

    assert len(environment.submitted) == 1


def test_the_kpath_example_sets_the_chosen_source(environment):
    from aiida import orm

    run_example("02_custom_kpath.py", environment.settings, {"KPATH_MODE": "explicit"})

    (_, inputs), = environment.submitted
    assert "bands_kpoints" in inputs and "kpoint_path" not in inputs
    assert isinstance(inputs["bands_kpoints"], orm.KpointsData) and len(inputs["bands_kpoints"].get_kpoints()) == 30


def test_the_plot_example_reads_the_bands(aiida_profile, tmp_path, monkeypatch):
    import runpy

    import numpy as np
    from aiida import orm
    from aiida.common.links import LinkType

    from .example_harness import EXAMPLES

    def bands_data():
        data = orm.BandsData()
        kpoints = orm.KpointsData()
        kpoints.set_kpoints(np.zeros((5, 3)))
        data.set_kpointsdata(kpoints)
        data.set_bands(np.arange(10, dtype=float).reshape(5, 2))
        return data.store()

    workchain = orm.WorkflowNode().store()
    for name in ("wannier90", "wannier90_qp"):
        bands_data().base.links.add_incoming(workchain, link_type=LinkType.RETURN, link_label=f"band_structures__{name}")

    namespace = runpy.run_path(str(EXAMPLES / "05_plot_results.py"), run_name="example")
    monkeypatch.chdir(tmp_path)

    assert set(namespace["bands"](workchain)) == {"wannier90", "wannier90_qp"}
    assert (tmp_path / namespace["plot_bands"](workchain)).exists()
