# Development

## Tests

```bash
pip install -e .[testing]
pytest -v
```

The tests build the inputs of the workchain and of every example and validate them against the process specifications; they do not
run Quantum ESPRESSO, Wannier90 or Yambo. The data regression files (`tests/workflows/protocols/...`) are rewritten with
`REGEN_DATA=1 pytest` if `pytest-regressions` is not installed (or `pytest --force-regen` with it) when the protocols change on purpose.

## Documentation

```bash
pip install -e .[docs]
mkdocs serve        # or: mkdocs build --strict
```

The code of the how-to pages is included from the example scripts (`--8<-- "examples/...:name"`), so it cannot drift from the tested code.

## Code style

```bash
pip install -e .[pre-commit]
pre-commit install
```
