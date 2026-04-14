"""
Training pipeline for the Fire Sale Graph Transformer.

Two-phase training procedure (Section 3, Eq. A.5):
  Phase 1: Self-supervised pretraining with Masked Autoencoder (MAE)
    - 50 epochs, 35% edge masking rate
  Phase 2: Supervised fine-tuning with hurdle-model trade prediction
    - Up to 500 epochs, early stopping (patience=5) on validation regression correlation
    - MAE head frozen

Optimizer: AdamW (Kingma 2014; Loshchilov & Hutter 2018)
  - Learning rate: 1e-3
  - Weight decay: 1e-2
  - Learning rate reduction on plateau

Walk-forward evaluation (Section 4.1):
  - Expanding window, retrain from scratch each year
  - Test on strictly future data
"""

import os
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional

from .model import FireSaleGraphTransformer, ModelConfig
from .data import HoldingsGraph
from .metrics import compute_regression_correlation, compute_auc, compute_r_squared

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training hyperparameters from Table 2 and footnote 7."""

    def __init__(
        self,
        # MAE pretraining
        mae_epochs: int = 50,
        mae_mask_rate: float = 0.35,
        # Supervised fine-tuning
        supervised_epochs: int = 500,
        early_stopping_patience: int = 5,
        # Optimizer
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        # LR scheduler
        lr_patience: int = 3,
        lr_factor: float = 0.5,
        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # Checkpointing
        checkpoint_dir: str = "checkpoints",
    ):
        self.mae_epochs = mae_epochs
        self.mae_mask_rate = mae_mask_rate
        self.supervised_epochs = supervised_epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.device = device
        self.checkpoint_dir = checkpoint_dir


class Trainer:
    """Manages the two-phase training procedure."""

    def __init__(
        self,
        model: FireSaleGraphTransformer,
        train_graphs: list[HoldingsGraph],
        val_graphs: list[HoldingsGraph],
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.train_graphs = train_graphs
        self.val_graphs = val_graphs
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def _make_optimizer(self, params) -> AdamW:
        return AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def phase1_pretrain(self) -> list[float]:
        """Phase 1: Self-supervised MAE pretraining.

        Train backbone + MAE head on masked position reconstruction.
        50 epochs, 35% edge masking rate.

        Returns:
            List of per-epoch average MAE losses.
        """
        logger.info("=== Phase 1: MAE Pretraining ===")
        self.model.train()

        # Optimize backbone + MAE head parameters
        params = list(self.model.backbone.parameters()) + list(self.model.mae_head.parameters())
        optimizer = self._make_optimizer(params)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min",
            patience=self.config.lr_patience,
            factor=self.config.lr_factor,
        )

        epoch_losses = []
        for epoch in range(self.config.mae_epochs):
            total_loss = 0.0
            num_graphs = 0

            for graph in self.train_graphs:
                graph = graph.to(self.device)
                optimizer.zero_grad()
                loss = self.model.forward_mae(graph, mask_rate=self.config.mae_mask_rate)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_graphs += 1

            avg_loss = total_loss / max(num_graphs, 1)
            epoch_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  MAE Epoch {epoch+1}/{self.config.mae_epochs}: "
                    f"loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        logger.info(f"  MAE pretraining complete. Final loss: {epoch_losses[-1]:.4f}")
        return epoch_losses

    def phase2_finetune(self) -> dict:
        """Phase 2: Supervised fine-tuning with trade prediction.

        Fine-tune backbone + trade prediction head. MAE head is frozen.
        Up to 500 epochs with early stopping (patience=5) based on
        validation regression correlation.

        Returns:
            Dict with training history:
              train_losses, val_losses, val_corrs, val_aucs, val_r2s, best_epoch
        """
        logger.info("=== Phase 2: Supervised Fine-tuning ===")

        # Freeze MAE head
        for param in self.model.mae_head.parameters():
            param.requires_grad = False

        # Optimize backbone + trade prediction head
        params = (
            list(self.model.backbone.parameters())
            + list(self.model.trade_head.parameters())
        )
        optimizer = self._make_optimizer(params)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max",
            patience=self.config.lr_patience,
            factor=self.config.lr_factor,
        )

        history = {
            "train_losses": [],
            "val_losses": [],
            "val_corrs": [],
            "val_aucs": [],
            "val_r2s": [],
            "best_epoch": 0,
        }

        best_val_corr = -float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.supervised_epochs):
            # ----- Training -----
            self.model.train()
            total_train_loss = 0.0
            num_train = 0

            for graph in self.train_graphs:
                graph = graph.to(self.device)
                if graph.trade_target is None:
                    continue

                optimizer.zero_grad()
                loss, _, _, _ = self.model.forward_trade_prediction(graph)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                num_train += 1

            avg_train_loss = total_train_loss / max(num_train, 1)
            history["train_losses"].append(avg_train_loss)

            # ----- Validation -----
            val_metrics = self._validate()
            history["val_losses"].append(val_metrics["loss"])
            history["val_corrs"].append(val_metrics["reg_corr"])
            history["val_aucs"].append(val_metrics["auc"])
            history["val_r2s"].append(val_metrics["r2"])

            scheduler.step(val_metrics["reg_corr"])

            # ----- Early stopping based on validation regression correlation -----
            if val_metrics["reg_corr"] > best_val_corr:
                best_val_corr = val_metrics["reg_corr"]
                patience_counter = 0
                history["best_epoch"] = epoch + 1
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            if (epoch + 1) % 25 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{self.config.supervised_epochs}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_corr={val_metrics['reg_corr']:.4f}, "
                    f"val_AUC={val_metrics['auc']:.4f}, "
                    f"val_R2={val_metrics['r2']:.4f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.6f}"
                )

            if patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"  Early stopping at epoch {epoch+1}. "
                    f"Best epoch: {history['best_epoch']} (corr={best_val_corr:.4f})"
                )
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        logger.info(
            f"  Fine-tuning complete. Best val corr: {best_val_corr:.4f} "
            f"at epoch {history['best_epoch']}"
        )

        # Unfreeze MAE head for potential future use
        for param in self.model.mae_head.parameters():
            param.requires_grad = True

        return history

    def _validate(self) -> dict:
        """Run validation and compute metrics."""
        self.model.eval()
        all_targets = []
        all_occurred = []
        all_c_hat = []
        all_m_hat = []
        all_y_hat = []
        total_loss = 0.0
        num_val = 0

        with torch.no_grad():
            for graph in self.val_graphs:
                graph = graph.to(self.device)
                if graph.trade_target is None:
                    continue

                loss, c_hat, m_hat, y_hat = self.model.forward_trade_prediction(graph)
                total_loss += loss.item()
                num_val += 1

                all_targets.append(graph.trade_target.cpu())
                all_occurred.append(graph.trade_occurred.cpu())
                all_c_hat.append(c_hat.cpu())
                all_m_hat.append(m_hat.cpu())
                all_y_hat.append(y_hat.cpu())

        if not all_targets:
            return {"loss": 0.0, "reg_corr": 0.0, "auc": 0.5, "r2": 0.0}

        targets = torch.cat(all_targets)
        occurred = torch.cat(all_occurred)
        c_hat = torch.cat(all_c_hat)
        m_hat = torch.cat(all_m_hat)

        # Regression correlation on non-zero trades
        nonzero = occurred > 0.5
        reg_corr = compute_regression_correlation(
            m_hat[nonzero], targets[nonzero]
        ) if nonzero.any() else 0.0

        # AUC for extensive margin
        auc = compute_auc(c_hat, occurred)

        # R-squared on non-zero trades
        r2 = compute_r_squared(
            m_hat[nonzero], targets[nonzero]
        ) if nonzero.any() else 0.0

        return {
            "loss": total_loss / max(num_val, 1),
            "reg_corr": reg_corr,
            "auc": auc,
            "r2": r2,
        }

    def train_full(self) -> dict:
        """Run the complete two-phase training procedure.

        Returns:
            Dict with phase1 and phase2 training histories.
        """
        mae_losses = self.phase1_pretrain()
        finetune_history = self.phase2_finetune()
        return {
            "phase1_mae_losses": mae_losses,
            "phase2": finetune_history,
        }

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.config.checkpoint_dir, "firesale_model.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": vars(self.model.config),
        }, path)
        logger.info(f"  Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"  Checkpoint loaded from {path}")


def walk_forward_evaluation(
    all_graphs: list[HoldingsGraph],
    model_config: Optional[ModelConfig] = None,
    train_config: Optional[TrainingConfig] = None,
    test_windows: Optional[list[tuple[int, int]]] = None,
) -> list[dict]:
    """Walk-forward out-of-sample evaluation (Section 4.1).

    The model is retrained from scratch each year on an expanding window
    of past data and tested on the subsequent year's quarters.

    From the paper: "The walk-forward evaluation spans six annual windows
    from 2015 to 2020."

    Args:
        all_graphs: List of HoldingsGraph objects sorted by quarter.
            Each must have .quarter set (e.g. "2015Q1").
        model_config: Model configuration.
        train_config: Training configuration.
        test_windows: List of (train_end_idx, test_end_idx) tuples.
            If None, inferred from quarters (yearly windows).

    Returns:
        List of per-window evaluation dicts with metrics.
    """
    model_config = model_config or ModelConfig()
    train_config = train_config or TrainingConfig()
    results = []

    if test_windows is None:
        # Infer yearly windows from quarter labels
        test_windows = _infer_yearly_windows(all_graphs)

    for window_idx, (train_end, test_end) in enumerate(test_windows):
        logger.info(f"\n=== Walk-forward window {window_idx + 1} ===")

        train_graphs = all_graphs[:train_end]
        test_graphs = all_graphs[train_end:test_end]

        if not test_graphs:
            continue

        # Hold out last year of training for validation
        val_split = max(1, len(train_graphs) - 4)  # ~4 quarters for validation
        val_graphs = train_graphs[val_split:]
        train_graphs_split = train_graphs[:val_split]

        logger.info(
            f"  Training quarters: {len(train_graphs_split)}, "
            f"Validation quarters: {len(val_graphs)}, "
            f"Test quarters: {len(test_graphs)}"
        )

        # Fresh model for each window
        model = FireSaleGraphTransformer(model_config)
        trainer = Trainer(model, train_graphs_split, val_graphs, train_config)
        trainer.train_full()

        # Evaluate on test set
        window_results = _evaluate_on_graphs(model, test_graphs, train_config.device)
        window_results["window"] = window_idx + 1
        window_results["train_quarters"] = len(train_graphs)
        window_results["test_quarters"] = len(test_graphs)
        results.append(window_results)

        logger.info(
            f"  Window {window_idx + 1} test results: "
            f"AUC={window_results['auc']:.4f}, "
            f"R2={window_results['r2']:.4f}, "
            f"Corr={window_results['reg_corr']:.4f}"
        )

    return results


def _evaluate_on_graphs(
    model: FireSaleGraphTransformer,
    graphs: list[HoldingsGraph],
    device: str,
) -> dict:
    """Evaluate model on a list of graphs."""
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    all_targets = []
    all_occurred = []
    all_c_hat = []
    all_m_hat = []
    all_y_hat = []

    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(dev)
            if graph.trade_target is None:
                continue
            inv_e, asset_e = model.backbone(graph)
            c_hat, m_hat, y_hat = model.trade_head(
                inv_e, asset_e, graph.edge_index_ia, graph.fsi
            )
            all_targets.append(graph.trade_target.cpu())
            all_occurred.append(graph.trade_occurred.cpu())
            all_c_hat.append(c_hat.cpu())
            all_m_hat.append(m_hat.cpu())
            all_y_hat.append(y_hat.cpu())

    if not all_targets:
        return {"auc": 0.5, "r2": 0.0, "reg_corr": 0.0, "comb_corr": 0.0, "num_observations": 0}

    targets = torch.cat(all_targets)
    occurred = torch.cat(all_occurred)
    c_hat = torch.cat(all_c_hat)
    m_hat = torch.cat(all_m_hat)
    y_hat = torch.cat(all_y_hat)
    nonzero = occurred > 0.5

    return {
        "auc": compute_auc(c_hat, occurred),
        "r2": compute_r_squared(m_hat[nonzero], targets[nonzero]) if nonzero.any() else 0.0,
        "reg_corr": compute_regression_correlation(m_hat[nonzero], targets[nonzero]) if nonzero.any() else 0.0,
        "comb_corr": compute_regression_correlation(y_hat, targets),
        "num_observations": int(targets.numel()),
    }


def _infer_yearly_windows(graphs: list[HoldingsGraph]) -> list[tuple[int, int]]:
    """Infer walk-forward windows from quarter labels.

    Groups by year, creates expanding train windows.
    """
    quarters = [g.quarter for g in graphs]
    years = sorted(set(q[:4] for q in quarters if q))

    windows = []
    for i, year in enumerate(years):
        if i < 2:  # Need at least 2 years for training
            continue
        train_end = sum(1 for q in quarters if q and q[:4] < year)
        test_end = sum(1 for q in quarters if q and q[:4] <= year)
        if train_end > 0 and test_end > train_end:
            windows.append((train_end, test_end))

    return windows
