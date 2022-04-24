"""Module with common data types."""
import enum


# AiiDA < 2.0 has no enum type, I use IntEnum to be compatible with int
# class Gw2wannier90SortType(enum.Enum):
class Gw2wannier90SortMode(enum.IntEnum):
    """Enumeration to indicate the sort mode of gw2wannier90."""

    # Sort amn/mmn/eig/spn/...
    # DEFAULT = 'default'
    DEFAULT = 0

    # In additional to DEFAULT, also sort chk
    # DEFAULT_AND_CHK = 'default_and_chk'
    DEFAULT_AND_CHK = 1

    # Do not sort amn/mmn/eig/spn/..., only add GW corrections to eig
    # NO_SORT = 'no_sort'
    NO_SORT = 2
