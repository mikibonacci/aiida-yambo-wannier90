# aiida-yambo-wannier90

Package devoted to the automatic calculation of G0W0 interpolated band structure of materials.

## Installation

Use the following commands to install the plugin::

    git clone https://github.com/aiidateam/aiida-yambo-wannier90 .
    cd aiida-yambo-wannier90
    pip install -e .  # also installs aiida, if missing (but not postgres)
    #pip install -e .[pre-commit,testing] # install extras for more features
    verdi quicksetup  # better to set up a new profile
    verdi plugin list aiida.calculations  # should now show your calculation plugins

Then use ``verdi code setup`` with the ``yambo_wannier90`` input plugin
to set up an AiiDA code for aiida-yambo-wannier90.

## Usage

A quick demo of how to submit a calculation::

    verdi daemon start         # make sure the daemon is running
    cd examples
    ./example_01.py -r        # submit test calculation
    verdi calculation list -a  # check status of calculation

If you have already set up your own aiida_yambo_wannier90 code using
``verdi code setup``, you may want to try the following command::

    yambo_wannier90-submit  # uses aiida_yambo_wannier90.cli

# Examples

For each examples, please update codes and structure.

1. [example01](./example_01.py): `Wannier90BandsWorkChain` for wannier90 bands

2. [example02](./example_02.py): `PwBaseWorkChain` for QE bands

3. [example03](./example_03.py): `YamboWorkflow` for QP energies

4. [example04](./example_04.py): `YppRestart` for `unsorted.eig`

5. [example05](./example_05.py): `Gw2wannier90Calculation` for sorted `eig`

6. [example06](./example_06.py): `YamboWannier90WorkChain` starting from `gw2wannier90` step

7. [example07](./example_07.py): `YamboConvergence` for Yambo automated convergence

8. [example08](./example_08.py): `YamboWannier90WorkChain` starting from `wannier90` step

9. [example09](./example_09.py): `YamboWannier90WorkChain` starting from scratch
