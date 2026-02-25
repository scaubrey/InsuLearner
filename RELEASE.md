# Release Checklist

Use this for each new PyPI/community release.

## 1. Preflight

- Confirm branch is clean for release changes.
- Confirm version in `pyproject.toml` is set correctly.
- Update `CHANGELOG.md` with release date and bullets.

## 2. Local Validation

Run from repo root:

```bash
ruff check .
pytest -q -m "not live"
```

Optional (requires credentials in `.env.test`):

```bash
pytest -q -m "live"
```

## 3. Packaging Validation

```bash
python -m pip install --upgrade pip build twine
python -m build
python -m twine check dist/*
```

Verify artifacts exist in `dist/`:

- source tarball (`.tar.gz`)
- wheel (`.whl`)

## 4. Publish

Using Poetry:

```bash
poetry publish --build
```

Or using Twine:

```bash
python -m twine upload dist/*
```

## 5. Post-release

- Create git tag for version (for example: `v0.2.0`).
- Create GitHub Release notes from `CHANGELOG.md`.
- Smoke test install in a fresh environment:

```bash
python -m venv /tmp/insulearner-smoke
source /tmp/insulearner-smoke/bin/activate
pip install insulearner
insulearner --help
```

## 6. Live Integration Check (Recommended)

- Run live parity test once after release using real credentials:

```bash
pytest -q -m "live" -k tidepool_nightscout_live_parity
```

- Confirm insulin/carbs parity and CIR estimate are within configured tolerances.
