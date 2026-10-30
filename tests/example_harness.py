"""Run an example script with test nodes instead of a cluster.

The examples are linear scripts with their settings on top (``PW_CODE = 'pw@mycluster'``, ``PARENT_PK = 1234``...). Here
they run for real with the settings which point to the database replaced by test nodes, any other setting replaced on
request (``overrides``), and ``aiida.engine.submit`` replaced by a function which does not submit but validates the
inputs against the process spec (what AiiDA does when submitting), so an API change which breaks an example fails a test.
"""
import pathlib
import re
from types import SimpleNamespace
from uuid import uuid4

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"

#: setting of the scripts -> entry point of the code (the plugin of the calculation it runs)
CODES = {
    "PW_CODE": "quantumespresso.pw",
    "PW2WANNIER90_CODE": "quantumespresso.pw2wannier90",
    "WANNIER90_CODE": "wannier90.wannier90",
    "PROJWFC_CODE": "quantumespresso.projwfc",
    "YAMBO_CODE": "yambo.yambo",
    "P2Y_CODE": "yambo.p2y",
    "YPP_CODE": "yambo.ypp",
    "GW2WANNIER90_CODE": "yambo_wannier90.gw2wannier90",
}


def make_environment(computer):
    """Codes (stored, so that they can be loaded by label), a parent calculation with a remote folder, and an ``ndb.QP``."""
    from aiida import orm
    from aiida.common.links import LinkType

    settings = {}
    for setting, entry_point in CODES.items():
        label = f"{entry_point}-{uuid4().hex[:8]}"
        orm.Code(
            label=label, input_plugin_name=entry_point, remote_computer_exec=[computer, "/bin/true"]
        ).store()
        settings[setting] = f"{label}@{computer.label}"

    calculation = orm.CalcJobNode(computer=computer).store()
    remote = orm.RemoteData(computer=computer, remote_path="/tmp/parent")
    remote.base.links.add_incoming(calculation, link_type=LinkType.CREATE, link_label="remote_folder")
    remote.store()
    qp_db = orm.SinglefileData.from_string("ndb.QP", filename="ndb.QP").store()
    nnkp = orm.SinglefileData.from_string("nnkp", filename="aiida.nnkp").store()
    eig = orm.SinglefileData.from_string("eig", filename="unsorted.eig").store()

    settings.update(
        {
            "PARENT_PK": remote.pk,
            "REMOTE_PK": remote.pk,
            "QP_DB_PK": qp_db.pk,
            "NNKP_PK": nnkp.pk,
            "UNSORTED_EIG_PK": eig.pk,
        }
    )
    return SimpleNamespace(settings=settings, submitted=[])


def install_submit(monkeypatch, submitted):
    """Replace `aiida.engine.submit` by a function validating the inputs of the process (nothing is run)."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage import get_manager

    # the plugin modules must be imported before: they should not get the replacement of `submit`
    import aiida_yambo_wannier90.workflows  # noqa: F401  # pylint: disable=unused-import
    import aiida_wannier90_workflows.workflows  # noqa: F401  # pylint: disable=unused-import

    def submit(process, **inputs):
        if hasattr(process, "process_class"):  # a builder
            process_class, inputs = process.process_class, process._inputs(prune=True)  # pylint: disable=protected-access
        else:
            process_class = process
        instantiated = instantiate_process(get_manager().get_runner(), process_class, **inputs)
        instantiated.close()
        submitted.append((process_class, inputs))
        return SimpleNamespace(pk=1, uuid="00000000-0000-0000-0000-000000000000")

    monkeypatch.setattr("aiida.engine.submit", submit)


def run_example(script, settings, overrides=None):
    """Execute the script with `settings` (those it defines) and `overrides` (all of them must be defined)."""
    path = EXAMPLES / script
    source = path.read_text()

    def replace(source, name, value, required):
        pattern = re.compile(rf"^{name} = .*$", re.M)
        if not pattern.search(source):
            assert not required, f"{script} does not define {name}"
            return source
        return pattern.sub(lambda _: f"{name} = {value!r}", source, count=1)

    for name, value in settings.items():
        source = replace(source, name, value, required=False)
    for name, value in (overrides or {}).items():
        source = replace(source, name, value, required=True)
    exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})  # pylint: disable=exec-used
