import html
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"


def _card(label: str, value: str, detail: str = "") -> str:
    """Render one dashboard summary card."""
    return (
        '<div class="card">'
        f'<span class="card-label">{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong>"
        f'<span class="card-detail">{html.escape(detail)}</span>'
        "</div>"
    )


def create_report(
    year: int = 2023,
    output_path: Path | None = None,
    include_audit: bool | None = None,
) -> Path:
    """Create a self-contained HTML report for a generated season prediction."""
    predictions_path = RESULTS_DIR / f"{year}_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_path}. Run the prediction step first."
        )

    output_path = output_path or RESULTS_DIR / f"f1_prediction_report_{year}.html"
    predictions = pd.read_csv(predictions_path)
    if include_audit is None:
        include_audit = (
            RESULTS_DIR / "uncertainty_calibration_summary.csv"
        ).exists() and (RESULTS_DIR / "permutation_importance_summary.csv").exists()
    valid_actual = predictions.dropna(subset=["Actual Position"])
    if len(valid_actual) > 3:
        correlation = spearmanr(
            valid_actual["Actual Position"], valid_actual["Predicted Position"]
        ).statistic
        correlation_text = f"{float(correlation):.3f}"
    else:
        correlation_text = "Pending"

    chart_name = f"predicted_vs_actual_{year}.png"
    chart_path = RESULTS_DIR / chart_name
    chart_markup = (
        f'<img src="{chart_name}" alt="Predicted versus actual {year} standings chart">'
        if chart_path.exists()
        else '<p class="muted">Run with <code>--visualise</code> to include the standings chart.</p>'
    )

    summary_path = RESULTS_DIR / "rolling_origin_summary.csv"
    if summary_path.exists():
        evaluation = pd.read_csv(summary_path)
        evaluation_markup = evaluation.to_html(
            index=False, classes="data-table", border=0, justify="left"
        )
    else:
        evaluation_markup = (
            '<p class="muted">Run <code>python scripts/evaluate.py</code> '
            "to include rolling-origin benchmark results.</p>"
        )

    paired_summary_path = RESULTS_DIR / "model_vs_naive_summary.csv"
    paired_chart_path = RESULTS_DIR / "model_vs_naive_by_season.png"
    if paired_summary_path.exists():
        paired_summary_markup = pd.read_csv(paired_summary_path).to_html(
            index=False, classes="data-table", border=0, justify="left"
        )
    else:
        paired_summary_markup = (
            '<p class="muted">Re-run the evaluation to include paired '
            "confidence intervals and season win/loss counts.</p>"
        )
    paired_chart_markup = (
        '<img src="model_vs_naive_by_season.png" '
        'alt="Per-season Spearman difference versus naïve baseline">'
        if paired_chart_path.exists()
        else ""
    )

    tier_summary_path = RESULTS_DIR / "tier_rolling_origin_summary.csv"
    tier_class_summary_path = RESULTS_DIR / "tier_rolling_origin_class_summary.csv"
    if tier_summary_path.exists() and tier_class_summary_path.exists():
        tier_summary = pd.read_csv(tier_summary_path)
        tier_class_summary = pd.read_csv(tier_class_summary_path)
        tier_markup = tier_summary.to_html(
            index=False, classes="data-table", border=0, justify="left"
        ) + tier_class_summary.to_html(
            index=False, classes="data-table", border=0, justify="left"
        )
    else:
        tier_markup = (
            '<p class="muted">Run <code>python scripts/evaluate.py</code> '
            "to include tier classification benchmark results.</p>"
        )

    error_season_path = RESULTS_DIR / "error_analysis_season_summary.csv"
    error_group_path = RESULTS_DIR / "error_analysis_group_summary.csv"
    if error_season_path.exists() and error_group_path.exists():
        error_seasons = pd.read_csv(error_season_path).sort_values(
            "mae", ascending=False
        )
        error_groups = pd.read_csv(error_group_path).sort_values("mae", ascending=False)
        error_season_markup = error_seasons.head(5).to_html(
            index=False, classes="data-table", border=0, justify="left"
        )
        error_group_markup = error_groups.to_html(
            index=False, classes="data-table", border=0, justify="left"
        )
        error_markup = (
            "<h3>Worst test seasons by MAE</h3>"
            + error_season_markup
            + "<h3>Error by driver/season type</h3>"
            + error_group_markup
        )
    else:
        error_markup = (
            '<p class="muted">Run <code>python scripts/error_analysis.py</code> '
            "to include error analysis.</p>"
        )

    calibration_summary_path = RESULTS_DIR / "uncertainty_calibration_summary.csv"
    calibration_bins_path = RESULTS_DIR / "uncertainty_calibration_bins.csv"
    if (
        include_audit
        and calibration_summary_path.exists()
        and calibration_bins_path.exists()
    ):
        calibration_markup = (
            pd.read_csv(calibration_summary_path).to_html(
                index=False, classes="data-table", border=0, justify="left"
            )
            + "<h3>Top-three probability calibration</h3>"
            + pd.read_csv(calibration_bins_path).to_html(
                index=False, classes="data-table", border=0, justify="left"
            )
        )
    else:
        calibration_markup = (
            '<p class="muted">Run <code>python scripts/model_audit.py</code> '
            "to include uncertainty calibration.</p>"
        )

    importance_path = RESULTS_DIR / "permutation_importance_summary.csv"
    if include_audit and importance_path.exists():
        importance_markup = (
            pd.read_csv(importance_path)
            .head(10)
            .to_html(index=False, classes="data-table", border=0, justify="left")
        )
    else:
        importance_markup = (
            '<p class="muted">Run <code>python scripts/model_audit.py</code> '
            "to include held-out permutation importance.</p>"
        )

    top_driver = str(predictions.iloc[0]["Driver"])
    top_team = str(predictions.iloc[0]["Team"])
    uncertainty_columns = [
        "Predicted Rank",
        "Driver",
        "Predicted Position",
        "Bootstrap Position SD",
        "Bootstrap Position P05",
        "Bootstrap Position P95",
        "Champion Probability",
        "Top 3 Probability",
        "Top 5 Probability",
    ]
    if set(uncertainty_columns).issubset(predictions.columns):
        uncertainty = predictions[uncertainty_columns].copy()
        for column in (
            "Champion Probability",
            "Top 3 Probability",
            "Top 5 Probability",
        ):
            uncertainty[column] = uncertainty[column].map(
                lambda value: f"{float(value):.0%}"
            )
        uncertainty_markup = uncertainty.to_html(
            index=False, classes="data-table", border=0, justify="left"
        )
        top_driver_detail = (
            f"{float(predictions.iloc[0]['Champion Probability']):.0%} champion "
            "bootstrap frequency; "
            f"{float(predictions.iloc[0]['Top 3 Probability']):.0%} top-3 frequency"
        )
    else:
        uncertainty_markup = (
            '<p class="muted">Rebuild the prediction artifact to include '
            "bootstrap uncertainty estimates.</p>"
        )
        top_driver_detail = top_team

    table_markup = predictions.to_html(
        index=False,
        classes="data-table",
        border=0,
        justify="left",
        table_id="prediction-table",
    )
    prediction_sort_script = """<script>
(() => {
  const select = document.getElementById("prediction-sort");
  const table = document.getElementById("prediction-table");
  if (!select || !table || !table.tBodies.length) return;

  const body = table.tBodies[0];
  const rows = Array.from(body.rows);
  const sortRows = () => {
    const column = Number(select.value);
    const descending = column === 12;
    rows.sort((left, right) => {
      const leftValue = Number.parseFloat(left.cells[column].textContent.trim());
      const rightValue = Number.parseFloat(right.cells[column].textContent.trim());
      if (Number.isNaN(leftValue) && Number.isNaN(rightValue)) return 0;
      if (Number.isNaN(leftValue)) return 1;
      if (Number.isNaN(rightValue)) return -1;
      return (leftValue - rightValue) * (descending ? -1 : 1);
    });
    rows.forEach((row) => body.appendChild(row));
  };

  select.addEventListener("change", sortRows);
})();
</script>"""
    cards = "".join(
        [
            _card("Point forecast champion", top_driver, top_driver_detail),
            _card(
                "Predicted position", f"{predictions.iloc[0]['Predicted Position']:.2f}"
            ),
            _card("Drivers ranked", str(len(predictions))),
            _card("Spearman vs actual", correlation_text),
        ]
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>F1 Championship Forecast | {year}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #667085;
      --paper: #f7f8fb;
      --panel: #ffffff;
      --red: #e10600;
      --blue: #1e41ff;
      --line: #e4e7ec;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 Inter, ui-sans-serif, system-ui, sans-serif; }}
    .shell {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }}
    .hero {{ background: linear-gradient(135deg, #141b2d, #273354); border-radius: 22px; color: white; padding: 34px; margin-bottom: 20px; }}
    .eyebrow {{ color: #ffb3ad; font-size: 12px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }}
    h1 {{ font-size: clamp(30px, 5vw, 54px); line-height: 1; margin: 10px 0 12px; letter-spacing: -.04em; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    .hero p {{ color: #d6dcf0; margin: 0; max-width: 720px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
    .card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 8px 28px rgba(23, 32, 51, .05); }}
    .card {{ min-height: 116px; padding: 18px; display: flex; flex-direction: column; gap: 5px; }}
    .card-label, .card-detail, .muted {{ color: var(--muted); }}
    .card-label {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }}
    .card strong {{ font-size: 24px; letter-spacing: -.03em; }}
    .card-detail {{ font-size: 13px; }}
    .panel {{ padding: 24px; margin-top: 20px; overflow-x: auto; }}
    .panel > p {{ color: var(--muted); margin-top: -5px; }}
    .data-table {{ border-collapse: collapse; width: 100%; min-width: 620px; }}
    .data-table th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; text-align: left; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid var(--line); padding: 11px 10px; white-space: nowrap; }}
    .data-table tr:first-child td {{ font-weight: 800; }}
    .data-table tr:hover td {{ background: #f8faff; }}
    .table-controls {{ display: flex; align-items: center; gap: 8px; margin-bottom: 12px; color: var(--muted); font-size: 13px; }}
    .table-controls select {{ border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; color: var(--ink); background: var(--panel); }}
    img {{ display: block; max-width: 100%; height: auto; border-radius: 12px; border: 1px solid var(--line); }}
    code {{ background: #eef1f7; border-radius: 5px; padding: 2px 5px; }}
    footer {{ color: var(--muted); font-size: 13px; margin-top: 22px; }}
    @media (max-width: 760px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }} .hero {{ padding: 25px; }} .panel {{ padding: 18px; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <div class="eyebrow">F1 Championship Forecast</div>
      <h1>{year} Drivers' Championship</h1>
      <p>A transparent view of the generated forecast, actual outcome comparison, and rolling-origin benchmark.</p>
    </header>
    <section class="cards">{cards}</section>
    <section class="panel">
      <h2>Predicted versus actual</h2>
      {chart_markup}
    </section>
    <section class="panel">
      <h2>Rolling-origin evaluation</h2>
      <p>Historical benchmark across chronological test seasons. Each test season is held out in full; lower RMSE is better, while higher R² and Spearman are better.</p>
      {evaluation_markup}
      <h3>Paired comparison with the naïve baseline</h3>
      <p>Confidence intervals resample whole test seasons. Win/loss counts compare each method with the baseline on exactly the same seasons.</p>
      {paired_chart_markup}
      {paired_summary_markup}
    </section>
    <section class="panel">
      <h2>Tier classification</h2>
      <p>Random Forest tier metrics averaged across the same chronological test seasons. Accuracy is the fraction of correctly classified drivers; macro F1 gives each tier equal weight.</p>
      {tier_markup}
    </section>
    <section class="panel">
      <h2>Where this model breaks</h2>
      <p>Error analysis uses the same walk-forward Random Forest predictions. Rookie/gap-returning labels and mid-season swaps are post-hoc diagnostic categories, not model inputs; 2022 is isolated as a predeclared regulation-change case study.</p>
      {error_markup}
    </section>
    <section class="panel">
      <h2>Bootstrap uncertainty</h2>
      <p>Each row reports the distribution from 100 season-level bootstrap Random Forest fits. Frequencies are the fraction of bootstrap rankings placing a driver champion, in the top 3, or in the top 5. P05–P95 describes model sensitivity, not a calibrated 90% prediction interval; historical coverage is reported below.</p>
      {uncertainty_markup}
    </section>
    <section class="panel">
      <h2>Uncertainty calibration</h2>
      <p>Historical coverage is measured only on untouched future seasons. Rolling conformal intervals use residuals available before each test season; Brier scores assess probability quality.</p>
      {calibration_markup}
    </section>
    <section class="panel">
      <h2>Held-out permutation importance</h2>
      <p>Importance is the increase in RMSE after permuting one feature in each held-out season. This replaces training-set impurity importance with an out-of-season measurement.</p>
      {importance_markup}
    </section>
    <section class="panel">
      <h2>Predicted order</h2>
      <div class="table-controls">
        <label for="prediction-sort">Sort rows by</label>
        <select id="prediction-sort">
          <option value="0">Predicted Rank</option>
          <option value="12">Champion Probability (highest first)</option>
          <option value="9">Bootstrap Position SD (lowest first)</option>
        </select>
      </div>
      {table_markup}
      {prediction_sort_script}
    </section>
    <footer>Generated locally from the pinned project pipeline. See MODEL_CARD.md for intended use and limitations.</footer>
  </main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    logger.info("Report saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    create_report()
