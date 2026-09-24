"""Pure HTML rendering for the season report."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportSections:
    cards: str
    chart: str
    evaluation: str
    paired_chart: str
    paired_summary: str
    tiers: str
    errors: str
    uncertainty: str
    calibration: str
    importance: str
    predictions: str
    prediction_sort_script: str


def render_report(year: int, sections: ReportSections) -> str:
    """Render a complete standalone report from already-prepared markup."""
    return f"""<!doctype html>
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
    <section class="cards">{sections.cards}</section>
    <section class="panel">
      <h2>Predicted versus actual</h2>
      {sections.chart}
    </section>
    <section class="panel">
      <h2>Rolling-origin evaluation</h2>
      <p>Historical benchmark across chronological test seasons. Each test season is held out in full; lower RMSE is better, while higher R² and Spearman are better.</p>
      {sections.evaluation}
      <h3>Paired comparison with the naïve baseline</h3>
      <p>Confidence intervals resample whole test seasons. Win/loss counts compare each method with the baseline on exactly the same seasons.</p>
      {sections.paired_chart}
      {sections.paired_summary}
    </section>
    <section class="panel">
      <h2>Tier classification</h2>
      <p>Random Forest tier metrics averaged across the same chronological test seasons. Accuracy is the fraction of correctly classified drivers; macro F1 gives each tier equal weight.</p>
      {sections.tiers}
    </section>
    <section class="panel">
      <h2>Where this model breaks</h2>
      <p>Error analysis uses the same walk-forward Random Forest predictions. Rookie/gap-returning labels and mid-season swaps are post-hoc diagnostic categories, not model inputs; 2022 is isolated as a predeclared regulation-change case study.</p>
      {sections.errors}
    </section>
    <section class="panel">
      <h2>Bootstrap uncertainty</h2>
      <p>Each row reports the distribution from 100 season-level bootstrap Random Forest fits. Frequencies are the fraction of bootstrap rankings placing a driver champion, in the top 3, or in the top 5. P05–P95 describes model sensitivity, not a calibrated 90% prediction interval; historical coverage is reported below.</p>
      {sections.uncertainty}
    </section>
    <section class="panel">
      <h2>Uncertainty calibration</h2>
      <p>Historical coverage is measured only on untouched future seasons. Rolling conformal intervals use residuals available before each test season; Brier scores assess probability quality.</p>
      {sections.calibration}
    </section>
    <section class="panel">
      <h2>Held-out permutation importance</h2>
      <p>Importance is the increase in RMSE after permuting one feature in each held-out season. This replaces training-set impurity importance with an out-of-season measurement.</p>
      {sections.importance}
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
      {sections.predictions}
      {sections.prediction_sort_script}
    </section>
    <footer>Generated locally from the pinned project pipeline. See MODEL_CARD.md for intended use and limitations.</footer>
  </main>
</body>
</html>
"""
