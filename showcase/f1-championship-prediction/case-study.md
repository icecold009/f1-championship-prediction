# F1 Championship Forecasting: case study

## Project overview

F1 Championship Forecasting is a research-oriented Python project that studies
whether historical driver and constructor performance can forecast the final
Formula 1 Drivers' Championship order. The project produces a ranked forecast,
derived standing tiers, uncertainty summaries, a rolling-origin evaluation, and
an HTML report.

The portfolio story is not that a machine-learning model beats a simple rule.
The strongest verified result is the opposite: across ten untouched future
seasons, the naive previous-season-order baseline remains stronger overall than
the fitted regressors. The project turns that negative result into a transparent,
reproducible artifact.

## Problem and audience

The prediction target is a driver's final championship position and a derived
tier such as Champion, Podium, Top 5, Top 10, Midfield, or Backmarker. The
intended audience is analysts, learners, recruiters, and technical reviewers
who want to inspect whether a forecasting claim survives a realistic evaluation
protocol.

The core problem is temporal leakage. If same-season results appear in the
features, a model can look accurate without representing a real pre-season
forecast. A second problem is methodological: a fitted model is not meaningful
unless it is compared with a strong, explicit baseline.

## Goals and constraints

My goals were to:

- construct pre-season-safe driver and constructor features;
- evaluate rankings and position errors on future seasons;
- expose uncertainty and failure modes;
- preserve raw-data and release provenance;
- make the result readable through a generated report.

The project is explicitly out of scope for live race-by-race prediction,
betting, or safety-critical decisions. Historical source data is downloaded
locally and is not committed. The published example is a report artifact, not a
production forecasting API.

## Solution and key user flow

The workflow is:

1. Download the Ergast-compatible Formula 1 CSV snapshot.
2. Validate required columns, minimum contents, and raw-data hashes.
3. Build one row per driver-season from prior driver statistics and prior
   constructor strength.
4. Shift the historical features into the next season and add cold-start flags.
5. Train and evaluate candidate regressors and tiers with chronological
   holdouts.
6. Generate a forecast CSV, charts, an HTML report, and a release manifest.

The report is the main review surface. A reviewer can see the 2023 point
forecast, compare predicted and actual standings, sort the driver table,
inspect paired baseline results, and read the uncertainty and error audits.

## Design and technical decisions

The most important decision was to hold out every test season in full. For a
test season N+1, training includes seasons through N only; there is no random
row split that mixes future and past drivers. Current-season rows are used to
identify entrants and form evaluation targets, while same-season outcomes are
not predictors.

The feature set includes prior average finish and grid position, racecraft
delta, wins, podiums, points rates, reliability, sprint points, and prior
constructor championship strength. The first observed constructor in a season
is used as a historical approximation of the season-opening team. The model
card flags this as a limitation and recommends an explicit pre-season entry
list.

The project evaluates Ridge, Gradient Boosting, Random Forest, and two naive
baselines. It also fits 100 season-level bootstrap Random Forests for empirical
model-sensitivity frequencies. Release manifests bind outputs to a code
revision, raw-data archive and table hashes, package versions, and artifact
paths.

## Implementation highlights

- src/data_processing.py creates leakage-safe, one-row-per-driver-season
  features and identifies completed seasons.
- src/model.py contains the regression, tier, walk-forward, baseline, and
  bootstrap logic.
- src/predict.py writes the forecast artifact for a selected season.
- src/report.py renders the HTML report with sortable predictions, paired
  comparisons, uncertainty, error analysis, and held-out importance.
- scripts/build_release.py and scripts/check_release.py package and validate
  the release bundle.
- tests/ covers feature construction, model behavior, artifact layout, report
  rendering, provenance, and release validation.

## Validation and results

The committed release manifest and live report show the same central benchmark:

| Method | Test seasons | Mean RMSE | Mean Spearman |
|---|---:|---:|---:|
| Naive previous-season final order | 10 | 3.728 | 0.821 |
| Random Forest + cold-start flags | 10 | 5.982 | 0.811 |
| Random Forest, history only | 10 | 6.189 | 0.807 |

The history-only Random Forest's mean Spearman delta versus the naive baseline
is -0.014, with a paired 95% interval from -0.057 to 0.029. It wins four
seasons and loses six on Spearman and loses eight of ten seasons on RMSE. The
cold-start ablation reduces Random Forest RMSE by 0.207 positions relative to
the history-only version, but it does not overturn the baseline conclusion.

For the 2023 example report, Max Verstappen is the point-forecast champion,
22 drivers are ranked, and the report shows Spearman correlation of 0.819
against the actual standings. That is a concrete example artifact, not a
generalization to all future seasons.

## Challenges and tradeoffs

The model's uncertainty is deliberately presented with a warning. Across 222
driver-seasons, the bootstrap P05-P95 interval covers the actual position only
34.2% of the time. Rolling conformal intervals reach 97.7% coverage but have a
mean width of 19.899 positions, which makes them weakly informative. Bootstrap
frequencies are therefore model-sensitivity summaries, not calibrated
real-world probabilities.

The clearest error slices are missing-history cases. Returning-after-gap
drivers have MAE 12.562 across 11 observations, while rookies have MAE 7.362
across 31 observations. Established returning drivers have MAE 3.138 across
180 observations. These results favor better pre-season context over more
complex model selection.

## What I learned

I learned to treat baselines, holdout design, provenance, and uncertainty
calibration as first-class product features. A model can produce a polished
ranking and still fail to beat a simple rule. Showing that clearly is more
valuable than optimizing a headline metric without an honest comparison.

The second lesson is that explanation has to travel with the artifact. The
HTML report includes the forecast, benchmark, paired comparison, failure
analysis, and uncertainty caveat so a reviewer does not have to infer the
limitations from source code alone.

## Future improvements

- Replace the first-observed-constructor approximation with an explicit
  pre-season entry list.
- Add leakage-safe prior-season features from qualifying and pit-stop data.
- Add per-season tier confusion matrices and class-support counts.
- Evaluate rookies and returning-after-gap drivers as a dedicated cold-start
  slice.
- Keep the previous-season total-points baseline as a model-selection gate.

## Technologies used

Python 3.12-3.13, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn,
pytest, pytest-cov, Ruff, GitHub Actions, and GitHub Pages.

## Evidence and status

- Source and methodology: README.md, MODEL_CARD.md, RELEASE.md.
- Committed report and visuals: docs/index.html,
  docs/predicted_vs_actual_2023.png, docs/model_vs_naive_by_season.png,
  and docs/release_manifest.json.
- Live report: https://icecold009.github.io/f1-championship-prediction/
  checked on 2026-08-10.
- Local verification on 2026-08-10: Ruff lint passed, Ruff format check passed,
  and 40 pytest tests passed with 77.50% total coverage.
- Local generated CSVs: present in results/ in the checked-out workspace;
  their generated status is tracked by the release manifest and they are
  ignored by Git.
- [NEEDS EVIDENCE] No evidence was found for production users, revenue,
  adoption, testimonials, or a live prediction API. Those claims are
  intentionally not made.
