# F1 Championship Forecasting: screenshot and evidence checklist

## Current evidence inventory

| Evidence | Status | Use | Source or verification |
|---|---|---|---|
| Published HTML report | Live-verified on 2026-08-10 | Slides 1, 4, 6, 7 | https://icecold009.github.io/f1-championship-prediction/ |
| 2023 predicted-versus-actual chart | Committed and visually inspected | Slides 1 and 4 | docs/predicted_vs_actual_2023.png |
| Per-season baseline comparison chart | Committed and visually inspected | Slide 6 | docs/model_vs_naive_by_season.png |
| Release manifest | Committed, source-verified | Slides 3 and 5 | docs/release_manifest.json |
| Forecast CSV | Local generated artifact | Slide 4 and case study | results/2023_predictions.csv; ignored by Git |
| Rolling-origin summary | Local generated artifact | Slide 6 | results/rolling_origin_summary.csv; ignored by Git |
| Paired baseline summary | Local generated artifact | Slide 6 | results/model_vs_naive_summary.csv; ignored by Git |
| Tier and error summaries | Local generated artifacts | Slide 7 | results/tier_rolling_origin_class_summary.csv and results/error_analysis_group_summary.csv; ignored by Git |
| Uncertainty calibration summary | Local generated artifact | Slide 7 | results/uncertainty_calibration_summary.csv; ignored by Git |
| Held-out importance summary | Local generated artifact | Slides 5 and 7 | results/permutation_importance_summary.csv; ignored by Git |
| CI and tests | Local-verified on 2026-08-10: 40 passed, 77.50% coverage | Case study status | .github/workflows/ci.yml, tests/, targeted elevated local run |

## Suggested screenshot set

These names are ready for a later capture pass. The existing committed charts
are already suitable evidence assets; the first two live-report captures would
add contextual UI around them.

| Suggested filename | Caption | Capture target | Evidence rule |
|---|---|---|---|
| screenshots/01-live-report-hero.png | 2023 forecast headline and predicted-versus-actual chart | Top of the live report | Capture only the actual published report; do not recreate the cards |
| screenshots/02-live-report-benchmark.png | Rolling-origin table and paired comparison with the naive baseline | Benchmark section of live report | Keep the season count and metric labels visible |
| screenshots/03-predicted-vs-actual-2023.png | Predicted versus actual standings for the 2023 example | Existing docs/predicted_vs_actual_2023.png | Reuse the committed asset when possible |
| screenshots/04-model-vs-naive-by-season.png | Per-season Spearman delta versus the naive baseline | Existing docs/model_vs_naive_by_season.png | Preserve legend and zero reference line |
| screenshots/05-live-report-uncertainty.png | Bootstrap sensitivity frequencies and calibration caveat | Uncertainty section of live report | Include the warning that these are not calibrated odds |
| screenshots/06-live-report-failure-modes.png | Error by driver type and season | Error-analysis section of live report | Keep sample counts visible |

## Capture checklist

- [x] Confirm the live report URL loads and the title is "F1 Championship Forecast | 2023".
- [x] Confirm the live report shows 22 ranked drivers, Max Verstappen as point
  forecast champion, and Spearman 0.819 versus the actual standings.
- [x] Confirm the rolling-origin table covers 10 test seasons and includes the
  naive previous-season-order baseline.
- [x] Confirm the paired comparison includes confidence intervals and win/loss
  counts.
- [x] Confirm uncertainty is labeled as bootstrap/model sensitivity rather than
  calibrated probability.
- [x] Visually inspect the two committed chart assets.
- [ ] Capture and commit the optional live-report screenshots listed above.
- [x] Re-run the local lint, format, and test gates on this showcase branch:
  Ruff lint passed, Ruff format check passed, and 40 pytest tests passed with
  77.50% total coverage.

## Claim boundaries

- The 0.821, 0.811, 0.807, calibration, and error-slice values are historical
  walk-forward results recorded in the repository's validated report bundle.
- The 2023 0.819 value is an example report result, not a future-season
  guarantee.
- No user, revenue, adoption, production, betting, or API claims are supported
  by the inspected repository.
- [NEEDS EVIDENCE] Any future claim about production usage, adoption, revenue,
  testimonials, or a live prediction API requires an external source or live
  product verification.
- If a future release regenerates results from a new raw snapshot, update the
  showcase's date, manifest commit, and cited values together.
