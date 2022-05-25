========
Tutorial
========

All the examples provided with the plugin are executables, and can be launched from the command line:

::
    cd examples
    ./example_01.py     # this creates the inputs and plots the builder
    ./example_01.py -r  # actual run

To run on your own machine, you have to change the input structure and codes. 

Example 1: running Wannier90BandsWorkChain for wannier90 bands
--------------------------------------------------------------

The example 2 computes QE bands. These are needed in order to compare with W90 DFT interpolated bands and
assess their accuracy. 

Example 2: running PwBaseWorkchain for QE bands
-----------------------------------------------

The example 2 computes QE bands. These are needed in order to compare with W90 DFT interpolated bands and
assess their accuracy. 

Example 3: running YamboWorkflow
--------------------------------

The example 3 is built to run a single YamboWorkflow workchain, as performed in the yambo_commensurate 
step of the main YamboWannier90 workchain of the plugin. As in all the other examples, the inputs are created
using the get_builder_from_protocol() method of the YamboWorkflow class. For more details, please have a
look at the documentation of the yambo-aiida plugin (https://aiida-yambo.readthedocs.io/en/master/).

Example 4: running YppRestart
-----------------------------

This example runs a ypp calculation using the YppRestart workchain. In particular, this is done in order
to create the unsorted.eig file needed by wannier90 to have eigenvalues on which interpolate GW bands. 
For more details, please have a look at the documentation of the yambo-aiida plugin 
(https://aiida-yambo.readthedocs.io/en/master/).

Example 5: running Gw2wannier90Calculation for sorted eig
---------------------------------------------------------


Example 6: running YamboWannier90WorkChain starting from `gw2wannier90` step
----------------------------------------------------------------------------


Example 7: running YamboConvergence
-----------------------------------

The example 7 concerns the convergence of GW results using the YamboConvergence workchain provided 
from the yambo-aiida plugin. In particular, the parameters "BndsRnXp", "GbndRnge", "NGsBlkXp" 
are converged simultaneously. 
For more details, please have a look at the documentation of the yambo-aiida plugin 
(https://aiida-yambo.readthedocs.io/en/master/).


Example 9: running YamboWannier90WorkChain
------------------------------------------

This last example shows how to run a complete `YamboWannier90WorkChain` from scratch for Silicon. 
You only have to modify the example to load the correct structure and your codes, and then the workchain takes
care to perform all the necessary steps needed to obtain interpolated G0W0 band structure.
Final result is shown in Fig. . 

.. image:: ./images/Silicon_full.png

You can reproduce the same plot by running both example 2, to obtain the QE bands, both example 9 (this one), and then 
use the command 

::

    aiida-yambo-wannier90 plot bands <pk_qe_bands> <pk_yambo_wannier_90_workchain> 