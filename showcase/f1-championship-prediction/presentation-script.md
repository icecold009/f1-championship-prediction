# F1 Championship Forecasting: spoken script

Target length: approximately 2 minutes.

I built F1 Championship Forecasting to answer a deceptively simple question:
can historical driver and constructor performance predict the final Drivers'
Championship order before a season begins?

The important constraint is the information boundary. A model cannot use the
target season's race outcomes to predict that same season. So the pipeline
turns historical race data into prior-season features, holds out each future
season intact, and compares fitted models with simple baselines.

The user-facing result is a generated HTML report. For the 2023 example, it
shows a ranked field of 22 drivers, a predicted-versus-actual chart, per-driver
bootstrap sensitivity fields, and a sortable table. It then exposes the
rolling-origin benchmark, error analysis, calibration audit, and held-out
permutation importance.

The headline result is a negative one, and that is the point. Across ten
untouched test seasons from 2016 through 2025, the naive previous-season-order
baseline reaches mean Spearman correlation of 0.821. The Random Forest with
cold-start flags reaches 0.811, and the history-only forest reaches 0.807. The
history-only model wins four seasons and loses six on rank correlation, while
the paired 95% interval for its average improvement is -0.057 to 0.029.

The project also tests whether its uncertainty is trustworthy. The bootstrap
position intervals cover the actual position only 34.2% of the time across 222
driver-seasons, so they are labeled as model-sensitivity frequencies rather
than calibrated prediction intervals. Error analysis shows why: rookies and
drivers returning after a gap have much larger errors because the model has
little prior history.

My main takeaway is methodological. A stronger-looking model is not
automatically a better forecast. The next improvements are explicit pre-season
entry lists, additional leakage-safe features, and better tier diagnostics,
while keeping the previous-season baseline as a model-selection gate.
