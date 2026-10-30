"""Tests of the ``YamboWannier90WorkChain``: the protocol builder gives a valid process and the steps prepare the right inputs."""
from types import SimpleNamespace

import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage import get_manager


@pytest.fixture
def workchain(generate_builder_inputs):
    """Instantiate (without running) the workchain from the protocol builder of silicon."""
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    builder = YamboWannier90WorkChain.get_builder_from_protocol(**generate_builder_inputs("Si"))
    process = instantiate_process(get_manager().get_runner(), YamboWannier90WorkChain, **builder._inputs(prune=True))
    yield process
    process.close()


def test_the_protocol_builder_gives_a_valid_process(workchain):
    assert "qp" in workchain.inputs.yambo_qp
    assert "yres" not in workchain.inputs.yambo_qp
    assert "ypp_QP" not in workchain.inputs
    assert "QP_subsets_dict" not in workchain.inputs.yambo_qp  # given by the user, or built by the step


def test_the_yambo_qp_and_convergence_inputs_do_not_share_their_namespaces(generate_builder_inputs):
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    builder = YamboWannier90WorkChain.get_builder_from_protocol(**generate_builder_inputs("Si"))

    assert builder.yambo_qp.qp.yambo is not builder.yambo.ywfl.qp.yambo
    assert builder.yambo_qp.qp.yambo.parameters is builder.yambo.ywfl.qp.yambo.parameters  # the nodes are shared


def _fake_converged_workflow(parameters):
    remote = orm.Dict({"remote": True})  # stands for the RemoteData of the converged run
    return SimpleNamespace(
        inputs=SimpleNamespace(qp=SimpleNamespace(yambo=SimpleNamespace(parameters=orm.Dict(parameters)))),
        outputs=SimpleNamespace(remote_folder=remote),
    )


@pytest.fixture
def prepared_qp_inputs(workchain, monkeypatch):
    """The inputs of the yambo QP step, with the previous steps replaced by fakes."""
    from aiida_yambo_wannier90.workflows import __init__ as module  # noqa: F401  pylint: disable=unused-import
    import aiida_yambo_wannier90.workflows as workflows

    converged = _fake_converged_workflow({"arguments": ["gw0"], "variables": {"BndsRnXp": [[1, 40], ""]}})
    monkeypatch.setattr(workflows, "get_yambo_converged_workchain", lambda _: converged)
    monkeypatch.setattr(workflows, "get_yambo_nscf", lambda _: SimpleNamespace(outputs=SimpleNamespace(retrieved=None)))
    monkeypatch.setattr(workflows, "get_output_explicit_kpoints", lambda **_: "dense")
    monkeypatch.setattr(workflows, "kmapper", lambda **kwargs: orm.List([[1, 1, kwargs["start_band"].value, kwargs["end_band"].value]]))

    workchain.ctx.current_structure = workchain.inputs.structure
    workchain.ctx.kpoints_w90 = "coarse"
    workchain.ctx.kpoints_gw = workchain.ctx.kpoints_gw_conv = "same"
    workchain.ctx.wkchain_yambo_conv = "converged-convergence"

    return workchain.prepare_yambo_qp_inputs(), converged


def test_the_yambo_qp_step_uses_the_converged_parameters_and_the_new_api(prepared_qp_inputs):
    inputs, converged = prepared_qp_inputs

    assert inputs.qp.yambo.parameters.get_dict() == converged.inputs.qp.yambo.parameters.get_dict()
    assert inputs.parent_folder is converged.outputs.remote_folder
    assert inputs.qp.yambo.settings.get_dict()["COPY_DBS"] is True
    assert inputs.qp.yambo.settings.get_dict()["INITIALISE"] is False
    assert "yres" not in inputs


def test_the_yambo_qp_step_computes_the_qp_of_the_wannier_bands(prepared_qp_inputs, workchain):
    inputs, _ = prepared_qp_inputs
    subsets = inputs.QP_subsets_dict.get_dict()

    w90_parameters = workchain.inputs.wannier90.wannier90.wannier90.parameters.get_dict()
    start = max(w90_parameters.get("exclude_bands", [0])) + 1
    assert subsets["explicit"] == [[1, 1, start, start + w90_parameters["num_bands"] - 1]]
    assert (subsets["qp_per_subset"], subsets["parallel_runs"]) == (50, 4)


def _builder(generate_builder_inputs, **kwargs):
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    return YamboWannier90WorkChain.get_builder_from_protocol(**{**generate_builder_inputs("Si"), **kwargs})


def _w90_parameters(builder):
    return builder.wannier90.wannier90.wannier90.parameters.get_dict()


def test_the_protocol_reaches_the_yambo_and_wannier90_stages(generate_builder_inputs):
    fast = _builder(generate_builder_inputs, protocol="fast")
    precise = _builder(generate_builder_inputs, protocol="precise")

    assert fast.yambo.ywfl.qp.yambo.parameters.get_dict() != precise.yambo.ywfl.qp.yambo.parameters.get_dict()
    assert _w90_parameters(fast)["num_bands"] != _w90_parameters(precise)["num_bands"]


def test_the_overrides_reach_their_stage(generate_builder_inputs):
    builder = _builder(
        generate_builder_inputs,
        overrides={
            "yambo": {"ywfl": {"qp": {"yambo": {"metadata": {"options": {"max_wallclock_seconds": 111}}}}}},
            "wannier90": {"wannier90": {"wannier90": {"parameters": {"num_iter": 7}}}},
            "wannier90_qp": {"wannier90": {"parameters": {"num_iter": 9}}},
        },
    )

    assert builder.yambo.ywfl.qp.yambo.metadata.options.max_wallclock_seconds == 111
    assert _w90_parameters(builder)["num_iter"] == 7
    assert builder.wannier90_qp.wannier90.parameters.get_dict()["num_iter"] == 9


def test_the_two_wannier90_runs_share_the_wannier_functions(generate_builder_inputs):
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    builder = _builder(generate_builder_inputs, wannier_projection_type=WannierProjectionType.SCDM)
    dft, qp = _w90_parameters(builder), builder.wannier90_qp.wannier90.parameters.get_dict()

    for key in ("num_wann", "num_bands", "exclude_bands", "auto_projections"):
        assert dft.get(key) == qp.get(key), key
    assert qp["auto_projections"] is True


def test_the_codes_of_the_caller_are_not_modified(generate_builder_inputs):
    inputs = generate_builder_inputs("Si")
    codes = dict(inputs["codes"])
    codes["pw"] = codes["pw"].store().full_label  # a label: it is loaded, in the copy

    _builder(generate_builder_inputs, codes=codes)

    assert isinstance(codes["pw"], str)


def test_missing_codes_are_reported(generate_builder_inputs):
    inputs = generate_builder_inputs("Si")
    codes = {key: value for key, value in inputs["codes"].items() if key != "ypp"}

    with pytest.raises(ValueError, match="missing"):
        _builder(generate_builder_inputs, codes=codes)


def test_results_exposes_the_outputs_of_the_stages_which_ran(workchain, monkeypatch):
    from aiida_wannier90_workflows.workflows import Wannier90BaseWorkChain, Wannier90OptimizeWorkChain
    from aiida_yambo.workflows.yambowf import YamboWorkflow

    exposed = []
    monkeypatch.setattr(workchain, "exposed_outputs", lambda node, cls, namespace: exposed.append((node, cls, namespace)) or {})
    monkeypatch.setattr(workchain, "out_many", lambda _: None)
    monkeypatch.setattr(workchain, "out", lambda *_: None)
    monkeypatch.setattr(workchain, "should_run_wannier90", lambda: False)
    # the bands which `results` copies to `band_structures`
    monkeypatch.setattr(type(workchain), "outputs", property(lambda _: {"wannier90_qp": {"interpolated_bands": "bands"}}))

    workchain.ctx.wkchain_yambo_qp = "qp"
    workchain.ctx.wkchain_wannier90 = "w90"
    workchain.ctx.wkchain_wannier90_qp = "w90qp"
    workchain.results()

    assert exposed == [
        ("qp", YamboWorkflow, "yambo_qp"),  # YamboWorkflow, not YamboRestart: the outputs of the namespace are its own
        ("w90", Wannier90OptimizeWorkChain, "wannier90"),
        ("w90qp", Wannier90BaseWorkChain, "wannier90_qp"),
    ]
