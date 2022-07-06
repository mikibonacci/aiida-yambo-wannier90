# aiida-yambo-wannier90

[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]

Plugin to combine Wannier90 interpolations with GW corrections computed by Yambo

## Features

* Wannier interpolation of GW band structure

## Repository contents

* [`examples/`](examples/): An example of how to submit workflows using this plugin

## Installation

```shell
pip install aiida-yambo-wannier90
verdi quicksetup  # better to set up a new profile
verdi plugin list aiida.workflows  # should now show the workflows in the plugins
```

## Usage

Here goes a complete example of how to submit a test calculation using this plugin.

A quick demo of how to submit a calculation:
```shell
verdi daemon start     # make sure the daemon is running
cd examples
./example_01.py        # run test calculation
verdi process list -a  # check record of calculation
```

## Development

```shell
git clone https://github.com/aiidaplugins/aiida-yambo-wannier90 .
cd aiida-yambo-wannier90
pip install --upgrade pip
pip install -e .[pre-commit,testing]  # install extra dependencies
pre-commit install  # install pre-commit hooks
pytest -v  # discover and run all tests
```

## License

MIT

[ci-badge]: https://github.com/aiidaplugins/aiida-yambo-wannier90/workflows/ci/badge.svg?branch=main
[ci-link]: https://github.com/aiidaplugins/aiida-yambo-wannier90/actions
[cov-badge]: https://coveralls.io/repos/github/aiidaplugins/aiida-yambo-wannier90/badge.svg?branch=main
[cov-link]: https://coveralls.io/github/aiidaplugins/aiida-yambo-wannier90?branch=main
[docs-badge]: https://readthedocs.org/projects/aiida-yambo-wannier90/badge
[docs-link]: http://aiida-yambo-wannier90.readthedocs.io/
[pypi-badge]: https://badge.fury.io/py/aiida-yambo-wannier90.svg
[pypi-link]: https://badge.fury.io/py/aiida-yambo-wannier90
