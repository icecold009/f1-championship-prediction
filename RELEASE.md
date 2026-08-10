# Release and operational checklist

This repository produces local, auditable report artifacts. It does not
automatically publish predictions to a public endpoint.

## Local release build

From a clean checkout:

```bash
pip install -r requirements.txt
# Fast local bundle (omits the slow historical audit)
python scripts/build_release.py --download --year 2023
python scripts/check_release.py --quick --year 2023

# Validated evidence bundle (required for publication/sharing)
python scripts/build_release.py --download --year 2023 --full-audit
python scripts/check_release.py --year 2023
```

The build regenerates features, models, predictions, the comparison chart, the
HTML report, regression and tier rolling-origin evaluation files, and
`results/release_manifest.json`. The manifest records the Git commit, Python
and package versions, data snapshot, row/season counts, evaluation summary,
and generated artifact paths.

The fast build records `full_audit: false` and is accepted only by
`check_release.py --quick`. The validated build records `full_audit: true` and
runs the slower historical model audit: 100 season-level bootstrap refits per
held-out season, rolling conformal coverage, Brier scores, and held-out
permutation importance. The validator also checks the manifest mode, current
Git commit, raw-table hashes and sizes, non-empty artifacts, prediction ranks,
and probability ranges.

## GitHub Actions artifact build

The manual **Build release report** workflow runs the same process on Ubuntu
and uploads the `results/` directory as a downloadable Actions artifact. It is
triggered from the GitHub Actions tab with **Run workflow** and does not push
generated data or model files back to the repository.

## Release checks

Before sharing a report:

- Confirm `python scripts/check_release.py --year YEAR` passes.
- Inspect the generated HTML report and chart.
- Review `release_manifest.json` for the intended commit and data snapshot.
- Confirm the rolling baseline comparison is included.
- Confirm paired confidence intervals and season win/loss counts are included.
- Confirm the tier accuracy, macro F1, and per-tier F1 tables are included.
- Confirm the bootstrap uncertainty table and run count are included.
- Confirm historical bootstrap/conformal coverage and Brier scores are included.
- Confirm held-out permutation importance replaces impurity importance.
- Confirm the “Where this model breaks” error-analysis tables are included.
- Confirm every raw CSV checksum matches `data/raw/data_manifest.json`.
- State that the report is historical analysis, not a live or betting system.

## Monitoring boundary

This project currently has artifact-level monitoring rather than live-service
monitoring. The release check catches missing inputs, stale/missing models,
malformed prediction files, and missing report artifacts. If this becomes a
hosted service, add request logging, model/data freshness checks, prediction
drift monitoring, and an explicit retraining schedule before production use.
