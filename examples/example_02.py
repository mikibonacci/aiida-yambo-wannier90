#!/usr/bin/env runaiida
"""Run a test calculation on localhost.

Usage: ./example_02.py
"""
import pathlib

from aiida import orm, engine

from aiida_yambo_wannier90.calculations.gw2wannier90 import Gw2wannier90Calculation

INPUT_DIR = pathlib.Path(__file__).absolute().parent / "input_files" / "example_02"


def main():

    code = orm.load_code('gw2wannier90@localhost')

    computer = orm.load_computer('localhost')
    # parent_folder = orm.RemoteData(remote_path=str(INPUT_DIR / "unsorted"), computer=computer)
    parent_folder = orm.load_node(139124)
    print(parent_folder)

    # nnkp = orm.SinglefileData(file=INPUT_DIR / "aiida.nnkp")
    nnkp = orm.load_node(139125)
    print(nnkp)

    # unsorted_eig = orm.SinglefileData(file=INPUT_DIR / "aiida.gw.unsorted.eig")
    unsorted_eig = orm.load_node(139126)
    print(unsorted_eig)

    # set up calculation
    inputs = {
        "code": code,
        "parent_folder": parent_folder,
        "nnkp": nnkp,
        "unsorted_eig": unsorted_eig,
        "metadata": {
            "description": "Test job submission with the aiida_yambo_wannier90 plugin",
        },
    }

    result = engine.submit(Gw2wannier90Calculation, **inputs)

    print(f"Submitted {result}")


if __name__ == "__main__":
    main()
