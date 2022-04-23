"""Fixtures for testing parsers."""
import pytest


@pytest.fixture(scope="session")
def filepath_parsers_fixtures(filepath_tests):
    """Return the absolute filepath of the `tests/parsers/fixtures` folder.

    .. warning:: if this file moves with respect to the `tests` folder, the implementation should change.

    :return: absolute filepath of `tests` folder which is the basepath for all test resources.
    """
    return filepath_tests / "parsers" / "fixtures"
