import numpy as np
import pytest

from src.model import assign_tier, get_spearman


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (1, "Champion"),
        (3, "Podium"),
        (5, "Top 5"),
        (10, "Top 10"),
        (15, "Midfield"),
        (16, "Backmarker"),
    ],
)
def test_assign_tier_boundaries(position, expected):
    assert assign_tier(position) == expected


def test_assign_tier_missing_position_is_unknown():
    assert assign_tier(None) == "Unknown"
    assert assign_tier(np.nan) == "Unknown"


def test_get_spearman_perfect_and_inverse_rankings():
    assert get_spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert get_spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
