# Contributing

Use [GitHub Issues](https://github.com/ZhaoXinCAAS/ChineseCabbage-YellowHeart-Phenotyping/issues)
to report reproducibility problems or propose changes. Include the commit, OS,
Python version, exact command, input layout, expected result, and actual output.
Use a bundled demo input or a minimal shareable example where possible. Remove
identifying local paths and avoid attaching restricted study images or workbooks.

## Local checks

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run the
same checks used by CI from the repository root.

```bash
uv sync --locked --extra dev --python 3.13
uv run --locked ruff check src tests
uv run --locked pyrefly check
uv run --locked pytest
```

Keep changes focused and document any change to input schemas, normalization,
mask generation, or grade boundaries. For a change to a manuscript analysis,
include the input used and the affected outputs so the impact can be checked.
Keep generated outputs under `results/` and study inputs under `data/`, or pass
paths outside the checkout. Include an update to `uv.lock` with dependency changes.
