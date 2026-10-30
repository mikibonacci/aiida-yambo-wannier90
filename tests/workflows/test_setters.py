"""The setters of the builder: each keeps the related inputs consistent, and the result is a valid process."""
import pytest

from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage import get_manager


@pytest.fixture
def builder(generate_builder_inputs):
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    return YamboWannier90WorkChain.get_builder_from_protocol(**generate_builder_inputs("Si"))


def _valid(builder):
    """The builder gives a valid process (the validators of the spec pass)."""
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    process = instantiate_process(get_manager().get_runner(), YamboWannier90WorkChain, **builder._inputs(prune=True))
    process.close()
    return True


def _dft(builder):
    return builder.wannier90.wannier90.wannier90


def test_set_wannier_mesh_changes_every_place_where_it_appears(builder):
    from aiida_yambo_wannier90.utils.setters import set_wannier_mesh

    set_wannier_mesh(builder, [5, 5, 5])

    assert _dft(builder).parameters.get_dict()["mp_grid"] == [5, 5, 5]
    assert builder.wannier90_qp.wannier90.parameters.get_dict()["mp_grid"] == [5, 5, 5]
    assert len(builder.wannier90.nscf.kpoints.get_kpoints()) == len(_dft(builder).kpoints.get_kpoints())  # nscf == w90
    assert len(_dft(builder).kpoints.get_kpoints()) == 125
    assert _valid(builder)


def test_set_gw_mesh(builder):
    from aiida_yambo_wannier90.utils.setters import set_gw_mesh

    set_gw_mesh(builder, [4, 4, 4], force=True)

    assert builder.GW_mesh.get_kpoints_mesh()[0] == [4, 4, 4]
    assert builder.kpoints_force_gw.value is True


def test_set_wannier_bands_keeps_the_two_runs_equal(builder):
    from aiida_yambo_wannier90.utils.setters import set_wannier_bands

    set_wannier_bands(builder, num_bands=20, num_wann=12, exclude_bands=[1, 2])

    dft = _dft(builder).parameters.get_dict()
    qp = builder.wannier90_qp.wannier90.parameters.get_dict()
    assert (dft["num_bands"], dft["num_wann"], dft["exclude_bands"]) == (18, 12, [1, 2])
    for key in ("num_bands", "num_wann", "exclude_bands"):
        assert qp[key] == dft[key]
    assert builder.wannier90.nscf.pw.parameters.get_dict()["SYSTEM"]["nbnd"] == 20


def test_set_projections_sets_both_runs_and_drops_auto_projections(builder):
    from aiida_yambo_wannier90.utils.setters import set_projections

    set_projections(builder, ["Si:sp3"])

    assert _dft(builder).projections.get_list() == ["Si:sp3"]
    assert builder.wannier90_qp.wannier90.projections.get_list() == ["Si:sp3"]
    assert "auto_projections" not in _dft(builder).parameters.get_dict()
    assert "auto_projections" not in builder.wannier90_qp.wannier90.parameters.get_dict()


def test_set_qp_subsets_is_validated(builder):
    from aiida_yambo_wannier90.utils.setters import set_qp_subsets

    set_qp_subsets(builder, qp_per_subset=20, parallel_runs=2, parallelism="auto", max_wallclock_seconds=600)
    assert builder.yambo_qp.QP_subsets_dict.get_dict() == {
        "qp_per_subset": 20,
        "parallel_runs": 2,
        "parallelism": "auto",
        "max_wallclock_seconds": 600,
    }
    assert _valid(builder)

    with pytest.raises(ValueError, match="Unknown keys"):
        set_qp_subsets(builder, parallel_run=2)
    with pytest.raises(ValueError, match="set by the workchain"):
        set_qp_subsets(builder, range_QP=5.0)


def test_set_resources_sets_the_step_everywhere_it_appears(builder):
    from aiida_yambo_wannier90.utils.setters import set_resources

    paths = set_resources(builder, "gw", nodes=2, mpi=8, threads=4, time=1234, queue="long", prepend_text="module load yambo")

    assert paths == ["yambo.ywfl.qp.yambo", "yambo_qp.qp.yambo"]
    for options in (builder.yambo.ywfl.qp.yambo.metadata.options, builder.yambo_qp.qp.yambo.metadata.options):
        assert dict(options.resources) == {"num_machines": 2, "num_mpiprocs_per_machine": 8, "num_cores_per_mpiproc": 4}
        assert options.max_wallclock_seconds == 1234 and options.queue_name == "long"
        assert options.prepend_text.splitlines()[0] == "export OMP_NUM_THREADS=4"
        assert "module load yambo" in options.prepend_text
    assert _valid(builder)


def test_set_resources_npool_and_errors(builder):
    from aiida_yambo_wannier90.utils.setters import set_resources

    set_resources(builder, "wannier_nscf", mpi=4, npool=2)
    assert builder.wannier90.nscf.pw.parallelization.get_dict()["npool"] == 2

    with pytest.raises(ValueError, match="unknown step"):
        set_resources(builder, "gw3", nodes=1)
    with pytest.raises(ValueError, match="only for the Quantum ESPRESSO"):
        set_resources(builder, "ypp", npool=2)


def test_set_resources_all_steps(builder):
    from aiida_yambo_wannier90.utils.setters import STEPS, set_resources

    changed = set_resources(builder, "all", time=3600)

    assert len(changed) >= len(STEPS)
    assert builder.gw2wannier90.metadata.options.max_wallclock_seconds == 3600


def test_start_from_removes_the_earlier_stages(builder, aiida_localhost):
    from aiida_yambo_wannier90.utils.setters import STAGES, start_from

    remote = orm.RemoteData(computer=aiida_localhost, remote_path="/tmp/x")
    start_from(builder, "wannier90_qp", remote_input_folder=remote)

    assert [stage for stage in STAGES if stage in builder] == ["wannier90_qp"]
    assert builder.wannier90_qp.wannier90.remote_input_folder is remote
    assert _valid(builder)


def test_start_from_checks_the_arguments(builder):
    from aiida_yambo_wannier90.utils.setters import start_from

    with pytest.raises(ValueError, match="unknown stage"):
        start_from(builder, "wannier")
    with pytest.raises(ValueError, match="takes"):
        start_from(builder, "ypp", parent_folder=1, qp_dbs=2)


def test_start_from_the_yambo_qp_with_a_mesh(builder):
    from aiida_yambo_wannier90.utils.setters import start_from

    start_from(builder, "yambo_qp", gw_mesh=[4, 4, 4])

    assert "yambo" not in builder and "yambo_qp" in builder
    assert builder.GW_mesh.get_kpoints_mesh()[0] == [4, 4, 4]
    assert _valid(builder)


def test_print_summary(builder, capsys):
    from aiida_yambo_wannier90.utils.setters import print_summary, set_bands_kpath, set_qp_subsets

    set_bands_kpath(builder, distance=0.05)
    set_qp_subsets(builder, 10, 3)
    text = print_summary(builder)

    assert capsys.readouterr().out.strip() == text
    assert "stages: yambo -> yambo_qp -> ypp -> wannier90 -> gw2wannier90 -> wannier90_qp" in text
    assert "seekpath, distance 0.05" in text
    assert "'qp_per_subset': 10" in text
    assert "mesh" in text and "num_wann" in text


def test_different_bands_in_the_two_wannier90_runs_are_rejected(builder):
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    parameters = builder.wannier90_qp.wannier90.parameters.get_dict()
    parameters["num_bands"] = parameters["num_bands"] + 1
    builder.wannier90_qp.wannier90.parameters = orm.Dict(parameters)

    with pytest.raises(ValueError, match="differs between"):
        instantiate_process(get_manager().get_runner(), YamboWannier90WorkChain, **builder._inputs(prune=True))
