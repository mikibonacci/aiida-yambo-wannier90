"""Setters of the inputs of the ``YamboWannier90WorkChain`` builder, so that nothing has to be edited by hand.

Every function acts on the builder (or a copy of its inputs) and keeps the related inputs consistent.
"""
import typing as ty

from aiida import orm
from aiida.engine import ProcessBuilder

from aiida_wannier90_workflows.utils.kpoints import create_kpoints_from_mesh
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints, set_num_bands
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain
from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain

from aiida_yambo_wannier90.calculations.functions.kpath import kpoints_from_path
from aiida_yambo_wannier90.workflows.validation import KPATH_INPUTS


def set_bands_kpath(  # pylint: disable=too-many-arguments
    builder: ProcessBuilder,
    *,
    path: ty.Optional[ty.Sequence[ty.Tuple[str, str]]] = None,
    point_coords: ty.Optional[ty.Dict[str, ty.Sequence[float]]] = None,
    kpoints: ty.Optional[orm.KpointsData] = None,
    explicit_kpoints: ty.Optional[orm.KpointsData] = None,
    distance: ty.Optional[float] = None,
) -> str:
    """Choose the band path of the workchain: exactly one of the following (or nothing: seekpath, default distance).

    :param path: segments of high-symmetry points, e.g. ``[("G", "X"), ("X", "W"), ("W", "G")]``, with ``point_coords``
        (the crystal coordinates of each label, in the cell of ``builder.structure``). Wannier90 interpolates the
        bands along the segments. A segment starting where the previous ended continues it.
    :param kpoints: the same as a labelled ``KpointsData`` of high-symmetry points.
    :param explicit_kpoints: a labelled ``KpointsData`` list: the bands are computed exactly at these points.
    :param distance: the reference distance (1/Å) of seekpath, which then chooses the path (and the primitive cell).
    :return: the mode that was set: ``"path"``, ``"kpoints"``, ``"explicit_kpoints"``, ``"distance"`` or ``"seekpath"``.
    """
    if (path is None) != (point_coords is None):
        raise ValueError("`path` and `point_coords` must be given together")
    given = {
        name: value
        for name, value in (
            ("path", path),
            ("kpoints", kpoints),
            ("explicit_kpoints", explicit_kpoints),
            ("distance", distance),
        )
        if value is not None
    }
    if len(given) > 1:
        raise ValueError(f"give only one of `path`, `kpoints`, `explicit_kpoints`, `distance`, got {sorted(given)}")

    for name in KPATH_INPUTS:
        if name in builder:
            delattr(builder, name)

    if not given:
        return "seekpath"
    if "path" in given:
        builder.kpoint_path = kpoints_from_path(builder.structure, path, point_coords)
    elif "kpoints" in given:
        _check_labels(kpoints, "kpoints")
        builder.kpoint_path = kpoints
    elif "explicit_kpoints" in given:
        _check_labels(explicit_kpoints, "explicit_kpoints")
        builder.bands_kpoints = explicit_kpoints
    else:
        builder.bands_kpoints_distance = orm.Float(distance)
    return next(iter(given))


def _check_labels(kpoints: orm.KpointsData, name: str) -> None:
    if not isinstance(kpoints, orm.KpointsData):
        raise TypeError(f"`{name}` must be a `KpointsData`, got {type(kpoints).__name__}")
    if kpoints.labels is None:
        raise ValueError(f"`{name}` must have `labels`")


#: the parameters of ``wannier90.x`` which must be the same in the first (DFT) and in the QP wannier90 runs, because
#: the QP run reads the matrices computed by the first one
WANNIER_SHARED_PARAMETERS = ("num_wann", "num_bands", "exclude_bands", "auto_projections")


def sync_wannier90_qp(builder: ty.Union[ProcessBuilder, dict]) -> None:
    """Copy what the QP wannier90 run shares with the first one (number of bands and Wannier functions, projections).

    Both runs must use the same ``num_wann``, ``num_bands``, ``exclude_bands`` and projections: only the eigenvalues
    change. It does nothing if one of the two runs is not in the builder.
    """
    if "wannier90" not in builder or "wannier90_qp" not in builder:
        return
    dft = builder["wannier90"]["wannier90"]["wannier90"]  # Optimize -> Base -> calculation
    qp = builder["wannier90_qp"]["wannier90"]
    if "parameters" not in dft or "parameters" not in qp:
        return
    dft_parameters = dft["parameters"].get_dict()
    qp_parameters = qp["parameters"].get_dict()
    for key in WANNIER_SHARED_PARAMETERS:
        if key in dft_parameters:
            qp_parameters[key] = dft_parameters[key]
        else:
            qp_parameters.pop(key, None)
    qp["parameters"] = orm.Dict(qp_parameters)
    for port in ("projections",):
        if port in dft:
            qp[port] = dft[port]
        elif port in qp:
            qp.pop(port)


############################### meshes ###############################


def set_wannier_mesh(builder: ProcessBuilder, mesh: ty.Sequence[int]) -> None:
    """Set the k-points mesh of the Wannier90 runs (the DFT one, with its scf/nscf, and the QP one).

    The mesh is the one of the cell of the calculations: if seekpath runs (no ``kpoint_path``/``bands_kpoints``), the
    primitive cell it finds. The workchain makes it commensurate with the GW mesh (see ``set_gw_mesh``).
    """
    kpoints = create_kpoints_from_mesh(builder.structure, list(mesh))
    if "wannier90" in builder:
        set_kpoints(builder["wannier90"], kpoints, process_class=Wannier90OptimizeWorkChain)
    if "wannier90_qp" in builder:
        set_kpoints(builder["wannier90_qp"], kpoints, process_class=Wannier90BaseWorkChain)


def set_gw_mesh(builder: ProcessBuilder, mesh: ty.Sequence[int], force: bool = False) -> None:
    """Set the GW k-points mesh, skipping the GW convergence of the mesh (``GW_mesh``).

    :param force: use this mesh also for Wannier90 (``kpoints_force_gw``) instead of looking for a coarser commensurate one.
    """
    builder.GW_mesh = create_kpoints_from_mesh(builder.structure, list(mesh))
    builder.kpoints_force_gw = orm.Bool(force)


############################### Wannier functions ###############################


def set_wannier_bands(
    builder: ProcessBuilder,
    num_bands: int,
    num_wann: ty.Optional[int] = None,
    exclude_bands: ty.Optional[ty.Sequence[int]] = None,
) -> None:
    """Set the number of bands and of Wannier functions, in both Wannier90 runs and in the nscf calculation.

    :param num_bands: the bands of the nscf calculation (``nbnd``), including the excluded ones.
    :param exclude_bands: bands excluded from the Wannierization, starting from 1.
    """
    set_num_bands(builder["wannier90"], num_bands, exclude_bands, process_class=Wannier90OptimizeWorkChain)
    if num_wann is not None:
        calculation = builder["wannier90"]["wannier90"]["wannier90"]
        parameters = calculation["parameters"].get_dict()
        parameters["num_wann"] = num_wann
        calculation["parameters"] = orm.Dict(parameters)
    sync_wannier90_qp(builder)


def set_projections(builder: ProcessBuilder, projections: ty.Sequence[str]) -> None:
    """Use explicit analytic projections in both Wannier90 runs, e.g. ``["Si:s", "Si:sp3"]`` (no ``auto_projections``)."""
    calculation = builder["wannier90"]["wannier90"]["wannier90"]
    calculation["projections"] = orm.List(list(projections))
    parameters = calculation["parameters"].get_dict()
    parameters.pop("auto_projections", None)
    calculation["parameters"] = orm.Dict(parameters)
    sync_wannier90_qp(builder)


############################### yambo ###############################


def set_qp_subsets(builder: ProcessBuilder, qp_per_subset: int = 50, parallel_runs: int = 4, **options) -> None:
    """Set how the QP calculation is split (the QP splitter of ``YamboWorkflow``), in ``yambo_qp``.

    The QPs themselves are the ones of the bands of the Wannier functions at the k-points of the coarse mesh: they are
    set by the workchain (``boundaries`` and ``range_QP`` are not used). ``options`` are the other keys of
    ``QP_subsets_dict`` (``parallelism='auto'``, ``resources``, ``max_wallclock_seconds``, ``queue_name``...),
    checked here.
    """
    from aiida_yambo.workflows.yambowf import validate_QP_subsets_dict

    subsets = {"qp_per_subset": qp_per_subset, "parallel_runs": parallel_runs, **options}
    validate_QP_subsets_dict(subsets)
    ignored = [key for key in ("boundaries", "range_QP", "explicit") if key in subsets]
    if ignored:
        raise ValueError(f"{ignored} are set by the workchain from the Wannier90 bands and k-points")
    builder.yambo_qp.QP_subsets_dict = orm.Dict(subsets)


############################### resources ###############################

#: the namespaces of the builder of each step (the first part of the path is the stage)
STEPS: ty.Dict[str, ty.Tuple[str, ...]] = {
    "gw_scf": ("yambo.ywfl.scf.pw", "yambo_qp.scf.pw"),
    "gw_nscf": ("yambo.ywfl.nscf.pw", "yambo_qp.nscf.pw"),
    "gw": ("yambo.ywfl.qp.yambo", "yambo_qp.qp.yambo"),
    "ypp": ("ypp.ypp",),
    "wannier_scf": ("wannier90.scf.pw",),
    "wannier_nscf": ("wannier90.nscf.pw",),
    "wannier_projwfc": ("wannier90.projwfc.projwfc",),
    "wannier_pw2wannier90": ("wannier90.pw2wannier90.pw2wannier90",),
    "wannier90": ("wannier90.wannier90.wannier90", "wannier90_qp.wannier90"),
    "gw2wannier90": ("gw2wannier90",),
}
#: the steps which run Quantum ESPRESSO (they have ``npool``)
PW_STEPS = ("gw_scf", "gw_nscf", "wannier_scf", "wannier_nscf")


def _namespace(builder, path: str):
    namespace = builder
    for name in path.split("."):
        if name not in namespace:
            return None
        namespace = namespace[name]
    return namespace


def set_resources(  # pylint: disable=too-many-arguments,too-many-locals
    builder: ProcessBuilder,
    step: str,
    *,
    nodes: ty.Optional[int] = None,
    mpi: ty.Optional[int] = None,
    threads: ty.Optional[int] = None,
    time: ty.Optional[int] = None,
    queue: ty.Optional[str] = None,
    account: ty.Optional[str] = None,
    prepend_text: ty.Optional[str] = None,
    custom_scheduler_commands: ty.Optional[str] = None,
    npool: ty.Optional[int] = None,
) -> ty.List[str]:
    """Set the job options of one step (everywhere it appears in the builder); the options not given are not changed.

    :param step: one of ``STEPS``: ``gw_scf``, ``gw_nscf``, ``gw`` (p2y and GW), ``ypp``, ``wannier_scf``, ``wannier_nscf``,
        ``wannier_projwfc``, ``wannier_pw2wannier90``, ``wannier90`` (both runs), ``gw2wannier90``;
        or ``"all"`` for all of them.
    :param nodes, mpi, threads: nodes, MPI tasks per node, OpenMP threads per task (``OMP_NUM_THREADS`` is set).
    :param time: walltime in seconds. :param npool: k-points pools of ``pw.x`` (Quantum ESPRESSO steps only).
    :return: the namespaces which were changed.
    """
    steps = list(STEPS) if step == "all" else [step]
    unknown = [name for name in steps if name not in STEPS]
    if unknown:
        raise ValueError(f"unknown step {unknown}, choose among {sorted(STEPS)} or 'all'")
    if npool is not None and any(name not in PW_STEPS for name in steps) and step != "all":
        raise ValueError(f"`npool` is only for the Quantum ESPRESSO steps {PW_STEPS}")

    changed = []
    for name in steps:
        for path in STEPS[name]:
            namespace = _namespace(builder, path)
            if namespace is None:
                continue
            options = namespace.metadata.options
            if any(value is not None for value in (nodes, mpi, threads)):
                resources = dict(options.resources) if "resources" in options else {}
                resources.setdefault("num_machines", 1)
                for key, value in (
                    ("num_machines", nodes),
                    ("num_mpiprocs_per_machine", mpi),
                    ("num_cores_per_mpiproc", threads),
                ):
                    if value is not None:
                        resources[key] = value
                options.resources = resources
            if threads is not None:
                options.prepend_text = _with_omp(options.prepend_text if "prepend_text" in options else "", threads)
            if time is not None:
                options.max_wallclock_seconds = time
            if queue is not None:
                options.queue_name = queue
            if account is not None:
                options.account = account
            if prepend_text is not None:
                options.prepend_text = (options.prepend_text if "prepend_text" in options else "") + "\n" + prepend_text
            if custom_scheduler_commands is not None:
                options.custom_scheduler_commands = custom_scheduler_commands
            if npool is not None and name in PW_STEPS:
                parallelization = namespace["parallelization"].get_dict() if "parallelization" in namespace else {}
                parallelization["npool"] = npool
                namespace["parallelization"] = orm.Dict(parallelization)
            changed.append(path)
    if not changed:
        raise ValueError(f"the step `{step}` is not in the builder")
    return changed


def _with_omp(text: str, threads: int) -> str:
    """The prepend text with ``OMP_NUM_THREADS`` set to ``threads`` (and nothing else changed)."""
    kept = [line for line in text.splitlines() if "OMP_NUM_THREADS" not in line]
    return "\n".join([f"export OMP_NUM_THREADS={threads}", *kept])


############################### stages ###############################

#: the stages in the order the workchain runs them
STAGES = ("yambo", "yambo_qp", "ypp", "wannier90", "gw2wannier90", "wannier90_qp")

#: what a workchain which starts at a stage needs from the previous ones: argument -> path in the builder
START_INPUTS: ty.Dict[str, ty.Dict[str, str]] = {
    "yambo_qp": {"parent_folder": "yambo_qp.parent_folder"},
    "ypp": {"parent_folder": "ypp.parent_folder", "qp_db": "ypp.ypp.QP_DB", "nnkp": "ypp.ypp.nnkp_file"},
    "wannier90": {"unsorted_eig": "gw2wannier90.unsorted_eig", "nnkp": "gw2wannier90.nnkp"},
    "gw2wannier90": {
        "parent_folder": "gw2wannier90.parent_folder",
        "unsorted_eig": "gw2wannier90.unsorted_eig",
        "nnkp": "gw2wannier90.nnkp",
    },
    "wannier90_qp": {"remote_input_folder": "wannier90_qp.wannier90.remote_input_folder"},
}


def start_from(builder: ProcessBuilder, stage: str, **parents) -> None:
    """Start the workchain at ``stage``: the earlier stages are removed and what they would have produced is given.

    ``stage`` is one of ``STAGES`` (``"yambo"`` is the beginning, nothing is removed). The arguments are the data of
    the previous stages (see ``START_INPUTS``), e.g.::

        start_from(builder, "ypp", parent_folder=gw_remote_folder, qp_db=merged_qp)
        start_from(builder, "yambo_qp", parent_folder=converged_yambo_remote_folder)

    For ``yambo_qp`` you can give ``gw_mesh=[n1, n2, n3]`` instead of a ``parent_folder`` to run it on that mesh.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage `{stage}`, choose among {STAGES}")
    for earlier in STAGES[: STAGES.index(stage)]:
        if earlier in builder:
            delattr(builder, earlier)

    accepted = dict(START_INPUTS.get(stage, {}))
    if stage == "yambo_qp":
        accepted["gw_mesh"] = None
    unknown = [name for name in parents if name not in accepted]
    if unknown:
        raise ValueError(f"`{stage}` takes {sorted(accepted)}, not {unknown}")
    for name, value in parents.items():
        if name == "gw_mesh":
            set_gw_mesh(builder, value)
            continue
        *namespaces, port = accepted[name].split(".")
        target = builder
        for namespace in namespaces:
            target = target[namespace]
        target[port] = value


############################### summary ###############################


def print_summary(builder: ProcessBuilder) -> str:
    """Print (and return) what the workchain will do: the stages, the band path, the meshes, the Wannier functions."""
    lines = [f"YamboWannier90WorkChain: {builder.structure.get_formula()}"]
    lines.append("stages: " + " -> ".join(stage for stage in STAGES if stage in builder))

    if "kpoint_path" in builder:
        lines.append("band path: the high-symmetry points of `kpoint_path`")
    elif "bands_kpoints" in builder:
        lines.append(f"band path: {len(builder.bands_kpoints.get_kpoints())} explicit k-points (`bands_kpoints`)")
    elif "bands_kpoints_distance" in builder:
        lines.append(f"band path: seekpath, distance {builder.bands_kpoints_distance.value} (primitive cell)")
    else:
        lines.append("band path: seekpath (primitive cell)")

    if "wannier90" in builder:
        calculation = builder["wannier90"]["wannier90"]["wannier90"]
        parameters = calculation["parameters"].get_dict()
        mesh = parameters.get("mp_grid")
        lines.append(
            f"wannier90: mesh {mesh}, num_wann {parameters.get('num_wann')}, num_bands {parameters.get('num_bands')}, "
            f"exclude_bands {parameters.get('exclude_bands')}, "
            + ("projections " + str(calculation["projections"].get_list()) if "projections" in calculation
               else f"auto_projections {parameters.get('auto_projections', False)}")
        )
    if "GW_mesh" in builder:
        lines.append(f"GW mesh: {builder.GW_mesh.get_kpoints_mesh()[0]} (forced for wannier90: {builder.get('kpoints_force_gw')})")
    if "yambo_qp" in builder and "QP_subsets_dict" in builder["yambo_qp"]:
        lines.append(f"QP splitter: {builder['yambo_qp']['QP_subsets_dict'].get_dict()}")

    resources = []
    for name, paths in STEPS.items():
        for path in paths:
            namespace = _namespace(builder, path)
            if namespace is None or "resources" not in namespace.metadata.options:
                continue
            options = namespace.metadata.options
            resources.append(
                f"  {path}: {dict(options.resources)}"
                + (f", {options.max_wallclock_seconds} s" if "max_wallclock_seconds" in options else "")
            )
    if resources:
        lines.append("resources:")
        lines.extend(resources)

    text = "\n".join(lines)
    print(text)
    return text
