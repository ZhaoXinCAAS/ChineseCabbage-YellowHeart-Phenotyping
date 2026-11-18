# Contributing

Use [GitHub Issues](https://github.com/ZhaoXinCAAS/ChineseCabbage-YellowHeart-Phenotyping/issues)
to report reproducibility problems or propose changes. Include the commit, OS,
Python version, exact command, input layout, expected result, and actual output.
Use a bundled demo input or a minimal shareable example where possible. Remove
identifying local paths and avoid attaching restricted study images or workbooks.

## Naming conventions

- Use lowercase `snake_case` for Python modules and scripts. Name modules for
  their responsibility and use a leading underscore for private module names.
- Use lowercase `kebab-case` for installed CLI commands. Prefer an action and
  its object for new commands. The public command names in `pyproject.toml` are
  stable even when they differ from the corresponding module name.
- In `scripts/manuscript/`, name scripts for the analysis or output they
  produce, with names such as `fisher_scores_table.py`. Keep manuscript figure
  and table numbers in documentation so filenames stay valid after renumbering.
  Document manuscript and package counterparts in the README's
  [workflow table](README.md#choose-a-workflow).
- Name test files `test_<module>.py` for a single module, or use a specific
  feature or workflow name when tests cover multiple modules.

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
