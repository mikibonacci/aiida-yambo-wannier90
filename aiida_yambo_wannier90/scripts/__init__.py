"""Executable scripts which the calculations of the plugin run on the computer (they are not AiiDA plugins)."""
import pathlib

#: Absolute path of ``gw2wannier90.py``: the ``remote_abs_path`` of the code of ``Gw2wannier90Calculation``
#: (the script must be on the computer where the calculation runs; copy it there if it is a remote one).
GW2WANNIER90_SCRIPT = pathlib.Path(__file__).resolve().parent / "gw2wannier90.py"
