===============
Getting started
===============

The aiida-yambo-wannier90 plugin is a package devoted to the automatic calculation of G0W0 interpolated
band structure of a given material, starting with very few inputs like the structure of the system.
The code automatically organizes and runs the needed calculations, starting from a G0W0 convergence test
and proceeding with the calculation of the wannierized wavefunctions and the interpolated band structures, 
both DFT and G0W0. The softwares used in the simulations are the quantumEspresso package, the Yambo Code and 
Wannier90. 

Installation
++++++++++++

Use the following commands to install the plugin::

    git clone https://github.com/aiidaplugins/aiida-yambo-wannier90 .
    cd aiida-yambo-wannier90
    pip install -e .  # also installs aiida, if missing (but not postgres)
    #pip install -e .[pre-commit,testing] # install extras for more features
    verdi quicksetup  # better to set up a new profile
    verdi plugin list aiida.calculations  # should now show your calculation plugins

Then use ``verdi code setup`` with the ``yambo_wannier90`` input plugin
to set up an AiiDA code for aiida-yambo-wannier90.

Usage
+++++

A quick demo of how to submit a calculation::

    verdi daemon start         # make sure the daemon is running
    cd examples
    ./example_01.py -r        # submit test calculation
    verdi calculation list -a  # check status of calculation

If you have already set up your own aiida_yambo_wannier90 code using
``verdi code setup``, you may want to try the following command::

    yambo_wannier90-submit  # uses aiida_yambo_wannier90.cli

Available calculations
++++++++++++++++++++++

.. aiida-calcjob:: Gw2wannier90Calculation
    :module: aiida_yambo_wannier90.calculations.gw2wannier90
