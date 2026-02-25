# InsuLearner: Estimating Insulin Pump Settings with Machine Learning
## Author: Cameron Summers
#### Author's Website: [www.CameronSummers.com](https://www.CameronSummers.com)

---

### _Warning_:
*_This code can have significant impact on insulin dosing.
There are no guardrails included here so it's possible to get poor
results in some circumstances.
Check with your doctor before making any changes to dosing settings._*

## Overview

This is the code underlying [my article](https://www.CameronSummers.com/how_I_calculate_my_sons_insulin_pump_settings_with_machine_learning),
where I describe the machine-learning approach used to estimate pump settings.

InsuLearner estimates:

- Carbohydrate Ratio (CIR)
- Basal Rate
- Insulin Sensitivity Factor (ISF)

Supported data sources:

- [Tidepool](https://www.tidepool.org)
- [Nightscout](https://nightscout.github.io/)

The model fits linear regression over aggregated carb/insulin windows and derives CIR + basal,
then uses CSF (`K`) and CIR to estimate ISF.

![Example settings plot](static/example_settings_plot_plus_aace.jpg)

## Dependencies

- Python `>=3.9,<4.0`

## Installation

```bash
pip install insulearner
```

Or for development:

```bash
git clone https://github.com/scaubrey/InsuLearner
cd InsuLearner
pip install -e .
```

## Usage (CLI)

The package installs a CLI entrypoint: `insulearner`.

### Tidepool Example

```bash
insulearner <your_tidepool_email> <your_tidepool_password> \
  --num_days 60 \
  --height_inches 72 \
  --weight_lbs 200 \
  --gender male
```

If you already know CSF:

```bash
insulearner <your_tidepool_email> <your_tidepool_password> --num_days 60 --CSF 4.2
```

### Nightscout Example

```bash
insulearner --source nightscout \
  --nightscout_url https://your-site.example.com \
  --nightscout_token <token-if-used> \
  --nightscout_api_secret <api-secret-if-used> \
  --num_days 30 \
  --CSF 12.5
```

Notes:

- `--source` defaults to `tidepool`.
- For `--source tidepool`, positional `tp_username tp_password` are required.
- For `--source nightscout`, `--nightscout_url` is required.

### Key CLI Options

- `--num_days`: number of days to analyze
- `--agg_period_window_size_hours`: aggregation window size in hours (default: `24`)
- `--agg_period_hop_size_hours`: hop size in hours (default: `24`)
- `--estimate_agg_boundaries`: estimate aggregation boundaries via autocorrelation-like logic

## Python API

```python
from InsuLearner import (
    TidepoolUser,
    NightscoutUser,
    TidepoolAPI,
    NightscoutAPI,
    analyze_settings_lr,
)
```

## Environment Variables for Tests (PyCharm-friendly)

Use `.env.test` in the repo root so you do not re-enter credentials each run.

1. Copy template:

```bash
cp .env.test.example .env.test
```

2. Fill values in `.env.test`.

3. Run tests from PyCharm or terminal. `tests/conftest.py` auto-loads `.env.test`.

Important:

- `.env.test` is git-ignored.
- `.env.test.example` is safe to commit.

## Test Suite

Run all non-live tests:

```bash
pytest -q -m "not live"
```

Run all tests including live (requires credentials):

```bash
pytest -q
```

Run only live parity tests:

```bash
pytest -q -m "live"
```

Run lint:

```bash
ruff check .
```

## CI

- PR/Push CI runs non-live tests only.
- Live API parity tests are split into a manual GitHub Actions workflow.

## Release Process

See [RELEASE.md](/Users/cameron/dev/InsuLearner/RELEASE.md) for the checklist.

## Acknowledgements

Special thanks to [Tidepool](https://www.tidepool.org) for serving the diabetes community.

## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)
