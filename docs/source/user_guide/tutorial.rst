========
Tutorial
========

This page can contain a simple tutorial for your code.

What we want to achieve
+++++++++++++++++++++++

The purpose of this plugin is to provide a workchain able to give you accurate G0W0 interpolated bands
performed with the Wannier90 and Yambo code, starting from scratch (i.e., only the structure and few
other inputs).

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
You only have to modify the example to load the correct structure, and then the workchain takes
care to perform all the necessary steps needed to obtain interpolated GW band structure.

The final result
+++++++++++++++++++++++

Some text
