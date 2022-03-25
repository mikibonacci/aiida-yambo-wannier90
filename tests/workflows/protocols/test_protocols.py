# pylint: disable=redefined-outer-name
"""Tests for the ``YamboWannier90WorkChain.get_builder_from_protocol`` method."""
import pytest

from aiida.engine import ProcessBuilder

# from aiida.plugins import WorkflowFactory


def test_get_available_protocols():
    """Test ``YamboWannier90WorkChain.get_available_protocols``."""
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    # YamboWannier90WorkChain = WorkflowFactory('yambo_wannier90')

    protocols = YamboWannier90WorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ["fast", "moderate", "precise"]
    assert all("description" in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``YamboWannier90WorkChain.get_default_protocol``."""
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    assert YamboWannier90WorkChain.get_default_protocol() == "moderate"


@pytest.mark.parametrize("structure", ("Si", "H2O", "GaAs", "BaTiO3"))
def test_default_protocol(
    generate_builder_inputs, data_regression, serialize_builder, structure
):
    """Test ``YamboWannier90WorkChain.get_builder_from_protocol`` for the default protocol."""
    from aiida_yambo_wannier90.workflows import YamboWannier90WorkChain

    inputs = generate_builder_inputs(structure)
    builder = YamboWannier90WorkChain.get_builder_from_protocol(**inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_list():
    """Test pseudo."""
    from aiida_pseudo.groups.family import (
        CutoffsPseudoPotentialFamily,
        PseudoDojoFamily,
        SsspFamily,
    )

    from aiida import orm

    pseudo_set = (PseudoDojoFamily, SsspFamily, CutoffsPseudoPotentialFamily)
    pseudo_family1 = orm.QueryBuilder().append(pseudo_set)
    allfam = pseudo_family1.all()
    raise ValueError(f"{allfam}")
    # raise ValueError(f"{allfam} {allfam[0][0]}")
