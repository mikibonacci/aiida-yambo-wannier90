"""K-path helpers: from a path given by the user (labels and coordinates) to the objects Wannier90 wants."""
import typing as ty

import numpy as np

from aiida import orm
from aiida.engine import calcfunction

from aiida_wannier90_workflows.utils.kpoints import get_path_from_kpoints


@calcfunction
def kpoint_path_dict(kpoints: orm.KpointsData) -> orm.Dict:
    """The ``kpoint_path`` ``Dict`` (``path`` and ``point_coords``) of a ``Wannier90Calculation``, from labelled ``kpoints``."""
    return get_path_from_kpoints(kpoints)


def kpoints_from_path(
    structure: orm.StructureData,
    path: ty.Sequence[ty.Tuple[str, str]],
    point_coords: ty.Dict[str, ty.Sequence[float]],
) -> orm.KpointsData:
    """A labelled ``KpointsData`` of the high-symmetry points of ``path``, as wannier90-workflows reads a ``kpoint_path``.

    :param structure: the structure (its cell defines the crystal coordinates of the points).
    :param path: the segments, e.g. ``[("G", "X"), ("X", "U"), ("K", "G")]``: a segment which starts where the previous
        one ended continues it, any other starts a new (disconnected) piece of the path.
    :param point_coords: the crystal coordinates of every label, e.g. ``{"G": [0, 0, 0], "X": [0.5, 0, 0.5], ...}``.
    :return: a ``KpointsData`` with ``labels``. Consecutive points are separated by one more point, so that a segment can
        be told from a break (this is the convention of ``get_path_from_kpoints``).
    """
    if len(path) == 0:
        raise ValueError("`path` must have at least one segment")
    missing = sorted({label for segment in path for label in segment} - set(point_coords))
    if missing:
        raise ValueError(f"`point_coords` has no coordinates for {missing}")

    points, labels = [], []

    def add(label, coords):
        labels.append((len(points), label))
        points.append(list(coords))

    previous_end = None
    for start, end in path:
        if start != previous_end:
            add(start, point_coords[start])
        middle = 0.5 * (np.array(point_coords[start], dtype=float) + np.array(point_coords[end], dtype=float))
        points.append(list(middle))
        add(end, point_coords[end])
        previous_end = end

    kpoints = orm.KpointsData()
    kpoints.set_cell_from_structure(structure)
    kpoints.set_kpoints(points, labels=labels)
    return kpoints
