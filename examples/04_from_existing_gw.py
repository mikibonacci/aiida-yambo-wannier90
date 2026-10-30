"""Start the ``YamboWannier90WorkChain`` from a calculation you already have, skipping the earlier stages.

The stages run in this order: yambo (GW convergence) -> yambo_qp -> ypp -> wannier90 -> gw2wannier90 -> wannier90_qp.
``start_from(builder, stage, ...)`` removes the earlier stages and sets what they would have produced (STAGE):

    'yambo_qp'      you have a converged GW calculation: give its remote folder (``PARENT_PK``), or just a GW mesh
                    (``GW_MESH``) to run the QP calculation on that mesh without the GW convergence
    'ypp'           you have the QP calculation: its remote folder (``PARENT_PK``) and the merged ``ndb.QP`` (``QP_DB_PK``)
    'wannier90_qp'  you have the last intermediate steps: the remote folder of the ``gw2wannier90`` calculation (``REMOTE_PK``)

Status: tested here (the inputs are valid); not run on a cluster.
Edit: the code labels, STAGE and the pks.
"""
from aiida import load_profile, orm
from aiida.engine import submit
from ase.build import bulk

from aiida_yambo_wannier90.utils.setters import print_summary, set_wannier_mesh, start_from
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
STAGE = 'yambo_qp'
PARENT_PK = 1234  # RemoteData of the GW calculation ('yambo_qp', 'ypp')
GW_MESH = None  # e.g. [4, 4, 4]: for 'yambo_qp' without PARENT_PK
QP_DB_PK = 1235  # the merged ndb.QP ('ypp')
REMOTE_PK = 1236  # RemoteData of the gw2wannier90 calculation ('wannier90_qp')
WANNIER_MESH = [4, 4, 4]

structure = orm.StructureData(ase=bulk('Si', 'diamond', a=5.43))

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
set_wannier_mesh(builder, WANNIER_MESH)

# --8<-- [start:start_from]
if STAGE == 'yambo_qp' and GW_MESH is not None:
    start_from(builder, 'yambo_qp', gw_mesh=GW_MESH)
elif STAGE == 'yambo_qp':
    start_from(builder, 'yambo_qp', parent_folder=orm.load_node(PARENT_PK))
elif STAGE == 'ypp':
    start_from(builder, 'ypp', parent_folder=orm.load_node(PARENT_PK), qp_db=orm.load_node(QP_DB_PK))
elif STAGE == 'wannier90_qp':
    start_from(builder, 'wannier90_qp', remote_input_folder=orm.load_node(REMOTE_PK))
else:
    raise ValueError(f'unknown STAGE {STAGE!r}')
# --8<-- [end:start_from]

print_summary(builder)

node = submit(builder)
print(f'YamboWannier90WorkChain from `{STAGE}` submitted, pk = {node.pk}')
