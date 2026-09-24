import html
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from src.atomic_artifacts import atomic_write_text
from src.report_renderer import ReportSections, render_report

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

    document = render_report(
        year,
        ReportSections(
            cards=cards,
            chart=chart_markup,
            evaluation=evaluation_markup,
            paired_chart=paired_chart_markup,
            paired_summary=paired_summary_markup,
            tiers=tier_markup,
            errors=error_markup,
            uncertainty=uncertainty_markup,
            calibration=calibration_markup,
            importance=importance_markup,
            predictions=table_markup,
            prediction_sort_script=prediction_sort_script,
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_path, document)
    logger.info("Report saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    create_report()
