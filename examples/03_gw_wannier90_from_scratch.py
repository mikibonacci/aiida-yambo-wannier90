"""GW-corrected Wannier-interpolated bands of silicon, from the structure: the whole ``YamboWannier90WorkChain``.

Flow:  seekpath -> GW convergence (``YamboConvergence``) -> commensurate GW mesh -> GW on the Wannier k-points (QP splitter)
       -> wannier90 postprocessing (nnkp) -> ypp (unsorted.eig) -> wannier90 (DFT) -> gw2wannier90 -> wannier90 with the GW energies

The protocol fills every stage; the setters below change what you usually need: the Wannier k-mesh, the QP splitter, the
resources of each step, the band path (see 02_custom_kpath.py). ``print_summary`` shows what will run. The GW
convergence uses the default parameters of aiida-yambo (see 05_convergence of its examples to change them with
``overrides={'yambo': ...}``).

Status: tested here (the inputs are valid); not yet run on a cluster with this version.
Edit: the code labels, the mesh, the resources.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo_wannier90.utils.setters import print_summary, set_qp_subsets, set_resources, set_wannier_mesh
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
PROTOCOL = 'fast'  # 'fast', 'moderate' or 'precise': the GW and the Wannier90 stages
WANNIER_MESH = [4, 4, 4]  # coarse mesh of the Wannier functions: the GW k-points must be commensurate with it
QP_PER_SUBSET = 50  # QP in each GW run of the QP splitter
PARALLEL_RUNS = 4  # GW runs at the same time

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 2. Inputs: the protocol, then the setters
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
builder = YamboWannier90WorkChain.get_builder_from_protocol(
    codes, structure, protocol=PROTOCOL, pseudo_family=PSEUDO_FAMILY,
)

set_wannier_mesh(builder, WANNIER_MESH)  # also in the QP run and in the nscf calculation
set_qp_subsets(builder, qp_per_subset=QP_PER_SUBSET, parallel_runs=PARALLEL_RUNS, parallelism='auto')

# --8<-- [start:resources]
# nodes=, mpi= (per node), threads=, time= (seconds), queue=, account=, npool= (Quantum ESPRESSO steps only)
set_resources(builder, 'gw_scf', mpi=8, npool=2)
set_resources(builder, 'gw_nscf', mpi=8, npool=2)
set_resources(builder, 'gw', nodes=1, mpi=8, threads=2, time=4 * 3600)
set_resources(builder, 'wannier_scf', mpi=8, npool=2)
set_resources(builder, 'wannier_nscf', mpi=8, npool=2)
set_resources(builder, 'wannier90', mpi=1, time=3600)
# set_resources(builder, 'all', queue='my_queue', account='my_account')   # the same for every step
# --8<-- [end:resources]

print_summary(builder)

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(builder)
print(f'YamboWannier90WorkChain submitted, pk = {node.pk}')
print(f'Monitor with:  verdi process status {node.pk}   or   verdi process report {node.pk}')

# ---------------------------------------------------------------------------
# 4. After it finishes: see 05_plot_results.py
# ---------------------------------------------------------------------------
# node.outputs.band_structures.wannier90     # the DFT interpolated bands
# node.outputs.band_structures.wannier90_qp  # the GW-corrected interpolated bands
