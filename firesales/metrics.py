"""
Evaluation metrics for the fire-sale prediction model.

From the paper (Section 4, Tables 3-4):
  - Classification AUC: extensive margin (trade vs. no-trade)
  - Regression R-squared: on non-zero trade targets
  - Regression correlation (Pearson): on non-zero targets
  - Combined correlation: sigma(c_hat) * m_hat vs. y (full hurdle output)
  - Spearman rank correlation: for systemic risk ranking (Section 4.4)
"""

import torch
import numpy as np


def compute_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Area Under the ROC Curve.

    Args:
        logits: [N] raw classification logits (before sigmoid)
        labels: [N] binary labels (0 or 1)

    Returns:
        AUC score (0.5 = random, 1.0 = perfect)
    """
    scores = logits.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()

    # Handle edge cases
    if len(np.unique(y)) < 2:
        return 0.5

    # Sort by score descending
    order = np.argsort(-scores)
    y_sorted = y[order]

    # Count positives and negatives
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Compute AUC via trapezoidal rule on ROC
    tp = 0.0
    fp = 0.0
    auc = 0.0
    prev_fp = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] >= 0.5:
            tp += 1
        else:
            fp += 1
            auc += tp

    return auc / (n_pos * n_neg)


def compute_r_squared(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute out-of-sample R-squared.

    R2 = 1 - SS_res / SS_tot

    Args:
        predictions: [N] predicted values
        targets: [N] true values

    Returns:
        R-squared (can be negative if model is worse than mean)
    """
    pred = predictions.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()

    if len(y) < 2:
        return 0.0

    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if ss_tot < 1e-10:
        return 0.0

    return 1.0 - ss_res / ss_tot


def compute_regression_correlation(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    """Compute Pearson correlation coefficient.

    Args:
        predictions: [N] predicted values
        targets: [N] true values

    Returns:
        Pearson r
    """
    pred = predictions.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()

    if len(y) < 2:
        return 0.0

    pred_centered = pred - pred.mean()
    y_centered = y - y.mean()

    numer = np.sum(pred_centered * y_centered)
    denom = np.sqrt(np.sum(pred_centered ** 2) * np.sum(y_centered ** 2))

    if denom < 1e-10:
        return 0.0

    return numer / denom


def compute_spearman_rank_correlation(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    """Compute Spearman rank correlation.

    Used for systemic risk metric evaluation (Section 4.4, Figure 7).

    Args:
        predictions: [N] predicted values
        targets: [N] true values

    Returns:
        Spearman rho
    """
    pred = predictions.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()

    if len(y) < 2:
        return 0.0

    # Rank
    pred_ranks = _rank(pred)
    y_ranks = _rank(y)

    return float(np.corrcoef(pred_ranks, y_ranks)[0, 1])


def _rank(x: np.ndarray) -> np.ndarray:
    """Assign ranks (average for ties)."""
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def compute_selling_intensity(
    y_hat: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight_w: torch.Tensor,
    num_investors: int,
    aum: torch.Tensor,
) -> torch.Tensor:
    """Compute predicted selling intensity per investor (Section 4.4, Eq. 12).

    S_hat_i = (1/AUM_i) * sum_a w_{ia} * max(-y_hat_{ia}, 0)

    Args:
        y_hat: [E] predicted trade quantities
        edge_index: [2, E] (investor_idx, asset_idx)
        edge_weight_w: [E] market-value position w_{ia}
        num_investors: I
        aum: [I] total AUM per investor

    Returns:
        [I] predicted selling intensity per investor
    """
    inv_idx = edge_index[0]
    selling = edge_weight_w * torch.clamp(-y_hat, min=0.0)

    total_selling = torch.zeros(num_investors, device=y_hat.device, dtype=y_hat.dtype)
    total_selling.scatter_add_(0, inv_idx, selling)

    return total_selling / (aum + 1e-10)


def compute_conformal_intervals(
    residuals_cal: torch.Tensor,
    predictions_test: torch.Tensor,
    alpha: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute split conformal prediction intervals (Section 4.7).

    Uses Vovk et al. (2005), Lei et al. (2018) methodology.
    Guarantees marginal coverage >= 1 - alpha under exchangeability.

    Args:
        residuals_cal: [N_cal] absolute residuals |y - y_hat| from calibration set
        predictions_test: [N_test] predictions on test set
        alpha: miscoverage level (default 0.1 for 90% coverage)

    Returns:
        (lower, upper): prediction interval bounds, each [N_test]
    """
    n = len(residuals_cal)
    # Quantile level for finite-sample coverage
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)

    q = torch.quantile(residuals_cal.float(), level)

    lower = predictions_test - q
    upper = predictions_test + q

    return lower, upper


def compute_locally_adaptive_conformal_intervals(
    residuals_cal: torch.Tensor,
    mad_cal: torch.Tensor,
    predictions_test: torch.Tensor,
    mad_test: torch.Tensor,
    alpha: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Locally adaptive conformal intervals (heteroscedastic).

    Normalizes residuals by a local scale estimate (MAD) to produce
    intervals that are wider for harder-to-predict edges.

    From Figure 9: "locally adaptive (heteroscedastic intervals via
    residual normalization)"

    Args:
        residuals_cal: [N_cal] absolute residuals from calibration set
        mad_cal: [N_cal] local scale estimates for calibration
        predictions_test: [N_test] predictions on test set
        mad_test: [N_test] local scale estimates for test set
        alpha: miscoverage level

    Returns:
        (lower, upper): adaptive prediction interval bounds
    """
    # Normalize calibration residuals
    normalized = residuals_cal / (mad_cal + 1e-10)

    n = len(normalized)
    level = np.ceil((1 - alpha) * (n + 1)) / n
    level = min(level, 1.0)

    q = torch.quantile(normalized.float(), level)

    # Scale back by test MAD
    width = q * mad_test
    lower = predictions_test - width
    upper = predictions_test + width

    return lower, upper
