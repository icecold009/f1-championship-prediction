import logging
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"


def create_visualisation(year: int = 2023) -> Path:
    """Create and save a predicted-versus-actual standings chart for a season."""
    predictions_path = RESULTS_DIR / f"{year}_predictions.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {predictions_path}. Run the prediction step first."
        )

    df = pd.read_csv(predictions_path)

    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(df))
    bar_width = 0.35

    ax.barh(
        [i + bar_width / 2 for i in x],
        df["Actual Position"],
        height=bar_width,
        label="Actual Position",
        color="#e10600",
        alpha=0.85,
    )
    ax.barh(
        [i - bar_width / 2 for i in x],
        df["Predicted Rank"],
        height=bar_width,
        label="Predicted Rank",
        color="#1e41ff",
        alpha=0.85,
    )

    ax.set_yticks(list(x))
    ax.set_yticklabels(df["Driver"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Championship Position (lower = better)")
    ax.set_title(f"{year} F1 Championship — Predicted vs Actual Standings", fontsize=13, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"predicted_vs_actual_{year}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Chart saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    create_visualisation()
