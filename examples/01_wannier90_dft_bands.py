"""Wannier90-interpolated DFT bands of silicon, with ``Wannier90BandsWorkChain`` (no GW).

Flow:  seekpath -> scf -> nscf -> wannier90 -> bands   (the ``aiida-wannier90-workflows`` workchain)

The first step of the GW@Wannier90 calculation (03_gw_wannier90_from_scratch.py), alone: it shows the Wannier functions and
the interpolation you will correct with GW. The mesh, the number of bands and the parallelization are changed
with the setters of ``aiida-wannier90-workflows``.

Status: tested here (the inputs are valid); the same inputs ran on a cluster with ``aiida-wannier90-workflows``.
Edit: the code labels, the mesh, the resources.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

load_profile()

from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.utils.kpoints import create_kpoints_from_mesh
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_kpoints, set_num_bands, set_parallelization
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain

# ---------------------------------------------------------------------------
# 1. Settings: edit them
# ---------------------------------------------------------------------------
PW_CODE = 'pw@mycluster'
PW2WANNIER90_CODE = 'pw2wannier90@mycluster'
WANNIER90_CODE = 'wannier90@mycluster'
PSEUDO_FAMILY = 'PseudoDojo/0.4/PBE/SR/standard/upf'  # norm-conserving, needed by Yambo later: `aiida-pseudo install pseudo-dojo`
KMESH = [4, 4, 4]  # of the Wannier90 interpolation
NUM_BANDS = 14  # bands of the nscf calculation (including the excluded ones)
NPOOL = 1
MPI = 8  # per node

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

# ---------------------------------------------------------------------------
# 2. Inputs
# ---------------------------------------------------------------------------
codes = {'pw': PW_CODE, 'pw2wannier90': PW2WANNIER90_CODE, 'wannier90': WANNIER90_CODE}
builder = Wannier90BandsWorkChain.get_builder_from_protocol(
    codes=codes,
    structure=structure,
    pseudo_family=PSEUDO_FAMILY,
    projection_type=WannierProjectionType.ANALYTIC,  # the projections of the pseudopotentials: sp3 on Si
)

set_kpoints(builder, create_kpoints_from_mesh(structure, KMESH), process_class=Wannier90BandsWorkChain)
set_num_bands(builder, NUM_BANDS, process_class=Wannier90BandsWorkChain)
set_parallelization(builder, {'npool': NPOOL, 'num_mpiprocs_per_machine': MPI})

# ---------------------------------------------------------------------------
# 3. Submit
# ---------------------------------------------------------------------------
node = submit(builder)
print(f'Wannier90BandsWorkChain submitted, pk = {node.pk}')

# ---------------------------------------------------------------------------
# 4. After it finishes
# ---------------------------------------------------------------------------
# node = orm.load_node(<PK>)
# bands = node.outputs.band_structure   # BandsData: the interpolated bands
# node.outputs.wannier90.remote_folder  # reuse: the first stage of 04_from_existing_gw.py
