"""Choose the band path of the GW@Wannier90 calculation: seekpath, a distance, your path, or an explicit list of k-points.

The path of the bands is set with ``set_bands_kpath``, one source at a time (KPATH_MODE):

    'seekpath'   nothing given: seekpath chooses the path (and the primitive cell)
    'distance'   seekpath, with the reference distance DISTANCE (1/Angstrom) between the points
    'path'       your high-symmetry path: segments of labels + their coordinates; wannier90 interpolates along it
    'kpoints'    the same, as a labelled ``KpointsData`` of the high-symmetry points
    'explicit'   an explicit list of labelled k-points: the bands are computed exactly at these points
                 (for example to compare point by point with the Quantum ESPRESSO bands)

With 'path', 'kpoints' and 'explicit' seekpath does not run: the coordinates are crystal coordinates of YOUR cell. A segment
which starts where the previous one ended continues it; any other starts a new piece of the path.

Status: tested here (the inputs are valid); not run on a cluster.
Edit: the code labels, KPATH_MODE and the path.
"""
import numpy as np
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo_wannier90.calculations.functions.kpath import kpoints_from_path
from aiida_yambo_wannier90.utils.setters import print_summary, set_bands_kpath
from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

load_profile()

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
PW_CODE = 'pw@mycluster'
PW2WANNIER90_CODE = 'pw2wannier90@mycluster'
WANNIER90_CODE = 'wannier90@mycluster'
PROJWFC_CODE = 'projwfc@mycluster'
YAMBO_CODE = 'yambo@mycluster'
P2Y_CODE = 'p2y@mycluster'
YPP_CODE = 'ypp@mycluster'
GW2WANNIER90_CODE = 'gw2wannier90@mycluster'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'
KPATH_MODE = 'path'
DISTANCE = 0.02  # 1/Angstrom, for 'distance'

# The path of 'path' and 'kpoints': Gamma - X, X - U | K - Gamma - L (crystal coordinates of the fcc primitive cell)
PATH = [('G', 'X'), ('X', 'U'), ('K', 'G'), ('G', 'L')]
POINT_COORDS = {
    'G': [0.0, 0.0, 0.0],
    'X': [0.5, 0.0, 0.5],
    'U': [0.625, 0.25, 0.625],
    'K': [0.375, 0.375, 0.75],
    'L': [0.5, 0.5, 0.5],
}

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 2. Inputs
# ---------------------------------------------------------------------------
codes = {
    'pw': PW_CODE,
    'pw2wannier90': PW2WANNIER90_CODE,
    'wannier90': WANNIER90_CODE,
    'projwfc': PROJWFC_CODE,
    'yambo': YAMBO_CODE,
    'p2y': P2Y_CODE,
    'ypp': YPP_CODE,
    'gw2wannier90': GW2WANNIER90_CODE,
}
builder = YamboWannier90WorkChain.get_builder_from_protocol(codes, structure, pseudo_family=PSEUDO_FAMILY)

# --8<-- [start:kpath]
if KPATH_MODE == 'seekpath':
    set_bands_kpath(builder)
elif KPATH_MODE == 'distance':
    set_bands_kpath(builder, distance=DISTANCE)
elif KPATH_MODE == 'path':
    set_bands_kpath(builder, path=PATH, point_coords=POINT_COORDS)
elif KPATH_MODE == 'kpoints':
    # a labelled KpointsData of the high-symmetry points: made here, but it can come from anywhere
    set_bands_kpath(builder, kpoints=kpoints_from_path(builder.structure, PATH, POINT_COORDS))
elif KPATH_MODE == 'explicit':
    # a labelled list of k-points: here 30 points from Gamma to X, with the labels of the ends
    points = np.linspace(POINT_COORDS['G'], POINT_COORDS['X'], 30)
    explicit = orm.KpointsData()
    explicit.set_cell_from_structure(builder.structure)
    explicit.set_kpoints(points, labels=[(0, 'G'), (len(points) - 1, 'X')])
    set_bands_kpath(builder, explicit_kpoints=explicit)
else:
    raise ValueError(f'unknown KPATH_MODE {KPATH_MODE!r}')
# --8<-- [end:kpath]

print_summary(builder)

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(builder)
print(f'YamboWannier90WorkChain ({KPATH_MODE} path) submitted, pk = {node.pk}')
