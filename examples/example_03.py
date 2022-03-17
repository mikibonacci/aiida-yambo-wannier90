#!/usr/bin/env runaiida
"""Run a ``YamboWannier90WorkChain``.

Usage: ./example_03.py
"""
from aiida import orm, engine

from aiida_wannier90_workflows.utils.workflows.builder import print_builder

from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain


def main():

    codes = {
        "pw": 'qe-git-pw@prnmarvelcompute5',
        "pw2wannier90": 'qe-git-pw2wannier90@prnmarvelcompute5',
        "projwfc": 'qe-git-projwfc@prnmarvelcompute5',
        "wannier90": 'wannier90-git-wannier90@prnmarvelcompute5',
        "yambo": 'yambo-5.0-yambo@prnmarvelcompute5',
        "p2y": 'yambo-5.0-p2y@prnmarvelcompute5',
        "ypp": 'yambo-5.0-ypp@prnmarvelcompute5',
        "gw2wannier90": 'gw2wannier90@prnmarvelcompute5',
    }

    # Si2 from wannier90/example23
    structure = orm.load_node(139524)

    builder = YamboWannier90WorkChain.get_builder_from_protocol(
        codes = codes,
        structure = structure,
    )

    print_builder(builder)

    # result = engine.submit(builder)

    # print(f"Submitted {result}")


if __name__ == "__main__":
    main()
