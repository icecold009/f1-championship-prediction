# F1 Championship Forecasting: presentation outline

Status: evidence-backed portfolio showcase for a validated example report. The
repository and local artifacts are source evidence; the GitHub Pages report was
also checked live on 2026-08-10. This is not a claim of a production forecasting
service or user adoption.

## Project inputs

- Project name: F1 Championship Forecasting
- Repository: https://github.com/icecold009/f1-championship-prediction
- Live example report: https://icecold009.github.io/f1-championship-prediction/
- Audience: general portfolio audience, including recruiters and technical reviewers
- Purpose: portfolio case study
- Tone: clear, professional, concise
- Length: 7 slides, approximately 2 minutes spoken
- Central narrative: a careful forecasting study found that a simple previous-season-order baseline beat the fitted regressors overall, and the project makes that negative result inspectable.

## Evidence labels

- **Source-verified:** stated in committed source or documentation.
- **Local artifact:** present in the checked-out repository but generated result files may be ignored by Git.
- **Live-verified:** observed in the published GitHub Pages report on 2026-08-10.
- **[NEEDS EVIDENCE]:** not established by the repository or live check; the
  missing fact is named where relevant.

## Slide 1 - F1 Championship Forecasting

**Main message:** The project tests whether historical driver and constructor
performance can forecast the final Drivers' Championship order without leaking
the target season's outcomes.

**On-slide points:**

- Leakage-safe pre-season forecasting study
- 2023 example forecast with uncertainty and actual-outcome comparison
- Strongest finding: the naive previous-season-order baseline wins overall

**Recommended visual:** ../../docs/predicted_vs_actual_2023.png, also visible
inside the live report.

**Speaker notes:** I built an evidence-first forecasting study for Formula 1.
The headline is intentionally not "machine learning wins." Across ten untouched
future seasons, repeating the previous season's order is a stronger overall
ranking baseline than the fitted regressors. The showcase is about making that
result reproducible and honest.

**Evidence:** README.md, ../../docs/predicted_vs_actual_2023.png,
../../docs/index.html, live-verified report URL.

## Slide 2 - Problem, audience, and why it matters

**Main message:** A pre-season ranking problem is easy to make look stronger
than it is if current-season outcomes leak into the features or if a simple
baseline is omitted.

**On-slide points:**

- Target: final driver championship position and tier
- Audience: analysts and reviewers comparing forecasting methods
- Constraints: only information available before the target season; seasons held out intact
- Why it matters: a plausible model is not useful if the evaluation is optimistic

**Recommended visual:** A simple before/after framing: "pre-season inputs" on
the left and "final standings" on the right, with a visible holdout boundary.

**Speaker notes:** The project is aimed at research and learning, not betting or
live race prediction. Each row represents a driver entering a season. The
challenge is to preserve a true pre-season information boundary while still
handling rookies, driver gaps, team changes, and incomplete historical records.

**Evidence:** README.md sections "Problem Framing", "Data", and "Features &
Approach"; MODEL_CARD.md sections "Intended use", "Out-of-scope use", and
"Prediction target and features".

## Slide 3 - Solution and primary workflow

**Main message:** The deliverable is an auditable pipeline from raw data to
forecast report, with provenance and chronological evaluation around the model.

**On-slide points:**

- Download and validate Ergast-compatible CSV tables
- Shift prior-season driver/team features into the forecast season
- Compare fitted models with explicit naive baselines
- Publish a report containing rankings, uncertainty, and failure modes

**Recommended visual:** This static workflow diagram.

~~~mermaid
flowchart LR
    A[Raw F1 CSV snapshot] --> B[Schema and checksum validation]
    B --> C[Prior-season feature engineering]
    C --> D[Chronological holdout evaluation]
    D --> E[Forecast and uncertainty artifact]
    E --> F[HTML report and GitHub Pages]
    D --> G[Baseline, error, calibration, and importance audits]
~~~

**Speaker notes:** The pipeline starts with a raw snapshot whose archive and
table hashes are recorded. Features are shifted by one season, while current
season rows are used only to identify entrants and construct evaluation targets.
The output is more than a prediction CSV: the release includes a report and
audits that show how the model compares, where it fails, and how cautious its
uncertainty estimates should be.

**Evidence:** scripts/download_data.py, src/data_processing.py,
scripts/evaluate.py, scripts/model_audit.py, scripts/error_analysis.py,
docs/release_manifest.json.

## Slide 4 - Product walkthrough: the report

**Main message:** The user-facing surface is a readable report that lets a
reviewer move from the headline forecast to the evidence behind it.

**On-slide points:**

- 2023 headline: Max Verstappen is the point-forecast champion
- 22 drivers are ranked and compared with actual standings
- Client-side sorting exposes uncertainty fields per driver
- Baseline, tier, error, calibration, and importance sections remain visible

**Recommended visual:** The live report at
https://icecold009.github.io/f1-championship-prediction/ plus the committed
../../docs/predicted_vs_actual_2023.png.

**Speaker notes:** The report opens with the 2023 forecast, predicted position,
number of ranked drivers, and Spearman correlation against the actual result.
From there, a reviewer can inspect the predicted-versus-actual chart, sort the
driver table, compare models season by season, and read the uncertainty audit.
The interface is intentionally transparent: the report labels bootstrap
frequencies as model sensitivity, not calibrated real-world odds.

**Evidence:** docs/index.html and live-verified page DOM, especially the
headline cards, prediction table, paired baseline section, and audit sections.

## Slide 5 - Technical approach and important decisions

**Main message:** The project prioritizes a defensible evaluation boundary over
model complexity.

**On-slide points:**

- Python, pandas, scikit-learn, matplotlib, pytest, and Ruff
- Prior driver form, reliability, points, and prior constructor strength
- Random Forest, Gradient Boosting, Ridge, and explicit baselines
- 100 season-level bootstrap fits for model-sensitivity frequencies
- Release manifests bind outputs to code, data hashes, and package versions

**Recommended visual:** A compact architecture view built from the workflow in
Slide 3, with the "evaluation gate" emphasized between features and report.

**Speaker notes:** The important implementation choice is not the Random Forest
itself; it is the time-aware protocol. Every test season is held out intact,
and the baseline is evaluated on the same seasons. Release manifests record
the code commit, raw-data checksum, processed artifact paths, and package
versions, which makes a published result traceable.

**Evidence:** src/model.py, src/predict.py, src/report.py,
tests/test_model.py, tests/test_professional_analysis.py,
tests/test_release.py, docs/release_manifest.json.

## Slide 6 - Results and validation

**Main message:** The study produces a useful negative result and explains its
uncertainty instead of hiding it behind a leaderboard.

**On-slide points:**

- Naive previous-season order: mean Spearman 0.821 across 10 test seasons
- Random Forest with cold-start flags: 0.811, delta -0.010 vs naive
- History-only Random Forest: 4 Spearman wins, 6 losses; RMSE wins 2 of 10
- 2023 example report: 0.819 Spearman against the actual standings

**Recommended visual:** ../../docs/model_vs_naive_by_season.png beside the
rolling-origin table in the live report.

**Speaker notes:** The fitted model is not presented as superior. The cold-start
flags lower Random Forest RMSE relative to the history-only version, but the
naive ordering remains stronger overall. The paired 95% interval for the
history-only Random Forest's Spearman delta is -0.057 to 0.029, so the project
does not claim a reliable improvement. The 2023 report is a concrete example,
not a substitute for the ten-season evaluation.

**Evidence:** README.md "Model Results"; MODEL_CARD.md "Reported results";
docs/release_manifest.json; live report tables; results/ CSVs when the local
release artifact is available.

## Slide 7 - Lessons, limitations, and next steps

**Main message:** The next improvement is better pre-season context and
cold-start evaluation, not more aggressive claims.

**On-slide points:**

- Bootstrap P05-P95 coverage is only 34.2% across 222 driver-seasons
- Returning-after-gap drivers have MAE 12.562; rookies have MAE 7.362
- Team-entry reconstruction still uses the first observed constructor
- Next: explicit entry lists, leakage-safe qualifying/pit-stop features, tier confusion matrices

**Recommended visual:** A two-column "what the model knows / where it breaks"
summary using the error-analysis and calibration tables.

**Speaker notes:** The most important lesson is that uncertainty must be
validated too. The bootstrap intervals are too narrow to be called calibrated
90% prediction intervals. The largest errors occur where history is missing.
The project therefore leaves a clear next-step list: improve entry-list
reconstruction, add safe pre-season features, and inspect tier support before
changing the classifier.

**Evidence:** MODEL_CARD.md "Uncertainty estimates" and "Where this model
breaks"; TODO.md; live report audit sections.

## Quality review

- One coherent story: problem -> leakage-safe pipeline -> report -> negative result -> next steps.
- Strongest visual appears on Slide 1 and returns as evidence on Slides 4 and 6.
- Metrics are tied to the ten-season walk-forward evaluation or the explicitly
  labeled 2023 example.
- The report is live-verified, but no production API, user count, revenue,
  adoption, or testimonial is claimed.
- [NEEDS EVIDENCE] Production users, adoption, revenue, testimonials, and a
  live prediction API were not established by the inspected evidence.
- The deck source is intentionally Markdown-first; a .pptx can be produced
  later from this outline if a presentation file is needed.
