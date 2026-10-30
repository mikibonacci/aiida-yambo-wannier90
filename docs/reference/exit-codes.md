# Exit codes

| Code | Label | Stage |
| --- | --- | --- |
| 401 | `ERROR_SUB_PROCESS_FAILED_SETUP` | the GW mesh and the Wannier90 mesh are not commensurate |
| 402 | `ERROR_SUB_PROCESS_FAILED_YAMBO_CONV` | `YamboConvergence` |
| 403 | `ERROR_SUB_PROCESS_FAILED_SETUP_KMESH` | the k-meshes |
| 404 | `ERROR_SUB_PROCESS_FAILED_WANNIER90_PP` | `wannier90 -pp` |
| 405 | `ERROR_SUB_PROCESS_FAILED_YAMBO_COMMENSURATE` | yambo on the commensurate mesh |
| 406 | `ERROR_SUB_PROCESS_FAILED_YAMBO_QP` | the QP calculation (see the exit codes of `YamboWorkflow`) |
| 407 | `ERROR_SUB_PROCESS_FAILED_YPP` | `ypp` |
| 408 | `ERROR_SUB_PROCESS_FAILED_WANNIER90` | the DFT wannier90 |
| 409 | `ERROR_SUB_PROCESS_FAILED_GW2WANNIER90` | `gw2wannier90` |
| 410 | `ERROR_SUB_PROCESS_FAILED_WANNIER90_QP` | wannier90 with the GW energies |

`verdi process report <pk>` of the workchain says which child failed. The failures of the Yambo steps are explained in the
documentation of `aiida-yambo` (restarts and exit codes).
