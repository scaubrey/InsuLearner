# Changelog

All notable changes to this project are documented in this file.

## [0.2.1] - 2026-04-02

### Changed

- README now documents Nightscout compatibility expectations and points users to open issues with redacted JSON samples when a deployment shape is unsupported.

### Fixed

- Nightscout `entries` fetching now supports sites that store entry timestamps as `date`, `mills`, or `created_at` instead of `dateString`.
- Nightscout `entries` fetching now falls back to client-side date filtering when a deployment allows unfiltered entry reads but returns no rows for server-side filtered requests.

## [0.2.0] - 2026-02-25

### Added

- Nightscout integration:
  - `NightscoutAPI`
  - `NightscoutUser`
  - `load_nightscout_user_data`
- CLI source selection via `--source {tidepool,nightscout}`.
- Nightscout credential flags:
  - `--nightscout_url`
  - `--nightscout_token`
  - `--nightscout_api_secret`
- Expanded test suite:
  - CLI smoke tests
  - API client tests
  - window/parsing tests
  - model stability tests
  - Tidepool/Nightscout fixture parity tests
  - credentialed live parity test (marked `live`)
- `pytest.ini` marker enforcement and marker taxonomy (`unit`, `integration`, `live`).
- `.env.test` loading for tests via `tests/conftest.py`.
- Public package exports in `InsuLearner/__init__.py`.
- Ruff configuration and dev dependency.
- GitHub Actions CI workflow for non-live tests.
- Release documentation (`RELEASE.md`).

### Changed

- Python support baseline moved to `>=3.9,<4.0`.
- CLI now emits source-specific validation errors and clearer runtime failure messages.
- README updated for Nightscout + testing workflow.

### Fixed

- Non-regression test path for synthetic dataset.
- Fixture parity behavior aligned with current Nightscout basal event handling.
