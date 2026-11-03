"""Knee-point budget detection and diminishing returns analyzer for allocation frontiers."""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class FrontierPoint:
    """Represents a discrete candidate budget point along an allocation frontier."""

    budget: float
    expected_net_profit: float
    expected_incremental_revenue: float = 0.0
    targeted_customers_count: int = 0


@dataclass(frozen=True)
class KneePointRecommendation:
    """Executive recommendation identifying the knee/elbow point on an allocation frontier."""

    recommended_budget: float
    expected_net_profit: float
    expected_incremental_revenue: float
    targeted_customers_count: int
    marginal_roi: float
    diminishing_returns_ratio: float
    explanation: str


def detect_frontier_knee_point(
    frontier_points: List[FrontierPoint],
) -> KneePointRecommendation:
    """Identifies the optimal knee/elbow budget point on an allocation profit frontier.

    Uses normalized geometric vector distance to the secant line (chord) connecting
    the first and last frontier points. Calculates marginal ROI degradation beyond
    the detected knee point and returns an executive recommendation.

    Args:
        frontier_points: Non-empty list of at least 3 FrontierPoint instances.

    Returns:
        KneePointRecommendation containing the optimal budget point, marginal ROI,
        diminishing returns ratio, and an executive explanation.

    Raises:
        ValueError: If frontier_points contains fewer than 3 points.
    """
    if frontier_points is None or len(frontier_points) < 3:
        raise ValueError(
            "frontier_points must contain at least 3 points to determine a knee point."
        )

    # Sort points deterministically by budget ascending
    sorted_points = sorted(
        frontier_points, key=lambda pt: (float(pt.budget), float(pt.expected_net_profit))
    )
    n = len(sorted_points)

    x = np.array([float(pt.budget) for pt in sorted_points], dtype=float)
    y = np.array([float(pt.expected_net_profit) for pt in sorted_points], dtype=float)

    # Normalize coordinates to [0, 1]
    x_range = float(np.max(x) - np.min(x))
    y_range = float(np.max(y) - np.min(y))

    x_norm = (x - np.min(x)) / x_range if x_range > 0 else np.zeros_like(x)
    y_norm = (y - np.min(y)) / y_range if y_range > 0 else np.zeros_like(y)

    # Chord vector from first to last normalized point
    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    chord_len = np.sqrt(dx**2 + dy**2)

    if chord_len > 0:
        # Perpendicular distance to the chord connecting (x_0, y_0) and (x_{n-1}, y_{n-1})
        distances = np.abs(dx * (y_norm - y_norm[0]) - dy * (x_norm - x_norm[0])) / chord_len
    else:
        distances = np.zeros_like(x)

    # Determine knee index: maximum perpendicular distance
    max_dist = float(np.max(distances))
    if max_dist > 0.0:
        knee_idx = int(np.argmax(distances))
    else:
        # On degenerate or perfectly linear frontiers, default to the median interior point
        knee_idx = n // 2

    knee_point = sorted_points[knee_idx]

    # Calculate pre-knee marginal ROI
    if knee_idx > 0:
        delta_budget_pre = float(x[knee_idx] - x[knee_idx - 1])
        delta_profit_pre = float(y[knee_idx] - y[knee_idx - 1])
        pre_roi = (
            delta_profit_pre / delta_budget_pre if delta_budget_pre > 0 else 0.0
        )
    else:
        pre_roi = 0.0

    # Calculate post-knee marginal ROI
    if knee_idx < n - 1:
        delta_budget_post = float(x[knee_idx + 1] - x[knee_idx])
        delta_profit_post = float(y[knee_idx + 1] - y[knee_idx])
        post_roi = (
            delta_profit_post / delta_budget_post if delta_budget_post > 0 else 0.0
        )
    else:
        post_roi = 0.0

    # Diminishing returns ratio = post_roi / pre_roi
    if pre_roi > 0.0:
        diminishing_returns_ratio = float(post_roi / pre_roi)
    else:
        diminishing_returns_ratio = 0.0

    marginal_roi = float(post_roi)

    # Construct executive explanation
    rec_budget = float(knee_point.budget)
    rec_profit = float(knee_point.expected_net_profit)

    if marginal_roi > 0.0:
        explanation = (
            f"Optimal budget knee point detected at ${rec_budget:,.2f} "
            f"(expected net profit: ${rec_profit:,.2f}). "
            f"Allocating beyond ${rec_budget:,.2f} yields less than "
            f"${marginal_roi:.2f} incremental profit per additional dollar spent "
            f"(diminishing returns ratio: {diminishing_returns_ratio:.2f})."
        )
    else:
        explanation = (
            f"Optimal budget knee point detected at ${rec_budget:,.2f} "
            f"(expected net profit: ${rec_profit:,.2f}). "
            f"Allocating beyond ${rec_budget:,.2f} yields zero or negative incremental profit "
            f"(marginal profit per dollar: ${marginal_roi:.2f})."
        )

    return KneePointRecommendation(
        recommended_budget=rec_budget,
        expected_net_profit=rec_profit,
        expected_incremental_revenue=float(knee_point.expected_incremental_revenue),
        targeted_customers_count=int(knee_point.targeted_customers_count),
        marginal_roi=marginal_roi,
        diminishing_returns_ratio=diminishing_returns_ratio,
        explanation=explanation,
    )

