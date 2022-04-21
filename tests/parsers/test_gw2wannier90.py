"""Tests for the ``Gw2wannier90Parser``."""


def test_gw2wannier90_out(data_regression, filepath_parsers_fixtures):
    """Test a minimal `gw2wannier90.py` calculation."""
    from aiida_yambo_wannier90.parsers.gw2wannier90 import parse_gw2wannier90_out

    filepath = (
        filepath_parsers_fixtures / "gw2wannier90" / "default" / "gw2wannier90.out"
    )
    with open(filepath) as handle:
        filecontent = handle.readlines()

    results = parse_gw2wannier90_out(filecontent)

    data_regression.check({"parameters": results.get_dict()})


def test_gw2wannier90_raw(data_regression, filepath_parsers_fixtures):
    """Test a minimal `gw2wannier90.py` calculation."""
    from aiida_yambo_wannier90.parsers.gw2wannier90 import parse_gw2wannier90_raw

    filepath = (
        filepath_parsers_fixtures / "gw2wannier90" / "raw" / "aiida.gw2wannier90.raw"
    )
    with open(filepath) as handle:
        filecontent = handle.readlines()

    results = parse_gw2wannier90_raw(filecontent)

    data_regression.check({"sort_index": results.get_array("sort_index").tolist()})


def test_gw2wannier90_raw_multilines(data_regression, filepath_parsers_fixtures):
    """Test a minimal `gw2wannier90.py` calculation."""
    from aiida_yambo_wannier90.parsers.gw2wannier90 import parse_gw2wannier90_raw

    filepath = filepath_parsers_fixtures / "gw2wannier90" / "raw" / "multilines.raw"
    with open(filepath) as handle:
        filecontent = handle.readlines()

    results = parse_gw2wannier90_raw(filecontent)

    data_regression.check({"sort_index": results.get_array("sort_index").tolist()})
