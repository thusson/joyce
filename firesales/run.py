"""
Main entry point for the Fire Sale Graph Transformer.

Usage:
    python -m firesales.run              # Run with synthetic data (smoke test)
    python -m firesales.run --help       # Show options

This script demonstrates:
  1. Graph construction from holdings data
  2. Two-phase training (MAE pretraining + supervised fine-tuning)
  3. Walk-forward evaluation
  4. Trade prediction and embedding extraction
"""

import argparse
import logging
import sys
import torch
import numpy as np

from .data import HoldingsGraph
from .model import FireSaleGraphTransformer, ModelConfig
from .train import Trainer, TrainingConfig, walk_forward_evaluation
from .metrics import (
    compute_r_squared,
    compute_regression_correlation,
    compute_auc,
    compute_selling_intensity,
    compute_conformal_intervals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def generate_synthetic_graph(
    num_investors: int = 100,
    num_assets: int = 200,
    avg_edges_per_investor: int = 20,
    fsi: float = 0.0,
    quarter: str = "2020Q1",
    with_targets: bool = True,
) -> HoldingsGraph:
    """Generate a synthetic holdings graph for testing.

    Creates random investor/asset features and a random bipartite graph.
    """
    rng = np.random.default_rng(hash(quarter) % 2**31)

    # Investor features
    inv_num = torch.tensor(rng.standard_normal((num_investors, 4)), dtype=torch.float32)
    inv_cat = torch.tensor(
        np.column_stack([
            rng.integers(0, 10, num_investors),
            rng.integers(0, 10, num_investors),
        ]),
        dtype=torch.long,
    )

    # Asset features
    asset_num = torch.tensor(rng.standard_normal((num_assets, 5)), dtype=torch.float32)
    asset_cat = torch.tensor(
        np.column_stack([
            rng.integers(0, 10, num_assets),
            rng.integers(0, 30, num_assets),
            rng.integers(0, 15, num_assets),
        ]),
        dtype=torch.long,
    )

    # Random bipartite edges
    num_edges = num_investors * avg_edges_per_investor
    inv_idx = torch.tensor(
        rng.integers(0, num_investors, num_edges), dtype=torch.long
    )
    asset_idx = torch.tensor(
        rng.integers(0, num_assets, num_edges), dtype=torch.long
    )
    # Deduplicate
    edge_set = set()
    unique_inv = []
    unique_asset = []
    for i, a in zip(inv_idx.tolist(), asset_idx.tolist()):
        if (i, a) not in edge_set:
            edge_set.add((i, a))
            unique_inv.append(i)
            unique_asset.append(a)

    edge_index = torch.stack([
        torch.tensor(unique_inv, dtype=torch.long),
        torch.tensor(unique_asset, dtype=torch.long),
    ])
    num_edges = edge_index.shape[1]

    # Random position sizes (market value and quantity)
    edge_w = torch.tensor(
        np.exp(rng.normal(13, 3, num_edges)), dtype=torch.float32
    )
    edge_q = torch.tensor(
        np.exp(rng.normal(8, 2, num_edges)), dtype=torch.float32
    )

    # Trade targets
    trade_target = None
    trade_occurred = None
    if with_targets:
        # ~40% of positions unchanged, rest have log-normal trades
        changed = rng.random(num_edges) > 0.4
        raw_targets = rng.normal(0, 0.5, num_edges) * changed
        raw_targets = np.clip(raw_targets, -5, 5)
        trade_target = torch.tensor(raw_targets, dtype=torch.float32)
        trade_occurred = torch.tensor(
            (np.abs(raw_targets) > 1e-9).astype(np.float32)
        )

    return HoldingsGraph(
        investor_num_features=inv_num,
        investor_cat_features=inv_cat,
        asset_num_features=asset_num,
        asset_cat_features=asset_cat,
        edge_index_ia=edge_index,
        edge_weight_w=edge_w,
        edge_weight_q=edge_q,
        trade_target=trade_target,
        trade_occurred=trade_occurred,
        fsi=fsi,
        quarter=quarter,
    )


def run_smoke_test():
    """Run a smoke test with synthetic data.

    Verifies:
      - Model construction and parameter count
      - MAE pretraining forward/backward pass
      - Trade prediction forward/backward pass
      - Embedding extraction
      - Metric computation
    """
    logger.info("=== Fire Sale Graph Transformer: Smoke Test ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # 1. Create model
    config = ModelConfig()
    model = FireSaleGraphTransformer(config)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    logger.info(
        f"Architecture: d_h={config.d_h}, d_e={config.d_e}, "
        f"L={config.num_layers}, S_A={config.num_heads}, "
        f"d_c={config.d_c}, dropout={config.dropout}"
    )

    # 2. Generate synthetic data
    logger.info("\nGenerating synthetic holdings graphs...")
    train_graphs = [
        generate_synthetic_graph(
            num_investors=50, num_assets=80, avg_edges_per_investor=10,
            fsi=np.random.normal(0, 1), quarter=f"{2010+i//4}Q{i%4+1}",
        )
        for i in range(8)
    ]
    val_graphs = [
        generate_synthetic_graph(
            num_investors=50, num_assets=80, avg_edges_per_investor=10,
            fsi=np.random.normal(0, 1), quarter=f"2012Q{i+1}",
        )
        for i in range(2)
    ]

    logger.info(
        f"Train graphs: {len(train_graphs)} quarters, "
        f"~{train_graphs[0].num_edges} edges each"
    )

    # 3. Train (reduced epochs for smoke test)
    logger.info("\nStarting two-phase training (reduced for smoke test)...")
    train_config = TrainingConfig(
        mae_epochs=3,
        supervised_epochs=5,
        early_stopping_patience=3,
        device=device,
    )
    trainer = Trainer(model, train_graphs, val_graphs, train_config)
    history = trainer.train_full()

    # 4. Predictions
    logger.info("\nGenerating predictions...")
    test_graph = generate_synthetic_graph(
        num_investors=50, num_assets=80, avg_edges_per_investor=10,
        fsi=2.0,  # stress scenario
        quarter="2013Q1",
    ).to(torch.device(device))

    y_hat, c_hat, m_hat = model.predict(test_graph, fsi=2.0)
    logger.info(
        f"Predictions for {test_graph.num_edges} edges: "
        f"y_hat range [{y_hat.min():.3f}, {y_hat.max():.3f}]"
    )

    # 5. Embeddings
    inv_e, asset_e = model.get_embeddings(test_graph)
    logger.info(
        f"Embeddings: investors {inv_e.shape}, assets {asset_e.shape}"
    )

    # 6. Metrics
    if test_graph.trade_target is not None:
        targets = test_graph.trade_target
        occurred = test_graph.trade_occurred
        nonzero = occurred > 0.5

        auc = compute_auc(c_hat.cpu(), occurred.cpu())
        r2 = compute_r_squared(m_hat[nonzero].cpu(), targets[nonzero].cpu())
        corr = compute_regression_correlation(m_hat[nonzero].cpu(), targets[nonzero].cpu())

        logger.info(f"Test AUC: {auc:.4f}")
        logger.info(f"Test R2 (nonzero): {r2:.4f}")
        logger.info(f"Test Corr (nonzero): {corr:.4f}")

        # Selling intensity
        aum = torch.ones(test_graph.num_investors, device=test_graph.investor_num_features.device)
        selling = compute_selling_intensity(
            y_hat, test_graph.edge_index_ia, test_graph.edge_weight_w,
            test_graph.num_investors, aum,
        )
        logger.info(f"Selling intensity: mean={selling.mean():.4f}, max={selling.max():.4f}")

        # Conformal intervals
        residuals = torch.abs(targets[nonzero] - m_hat[nonzero]).cpu()
        lower, upper = compute_conformal_intervals(
            residuals[:len(residuals)//2],
            m_hat[nonzero][len(residuals)//2:].cpu(),
            alpha=0.1,
        )
        logger.info(f"90% conformal interval width: {(upper - lower).mean():.4f}")

    logger.info("\nSmoke test passed.")


def main():
    parser = argparse.ArgumentParser(
        description="Fire Sale Graph Transformer (Clayton & Coppola, 2026)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", default=True,
        help="Run smoke test with synthetic data (default)",
    )
    parser.add_argument(
        "--d-h", type=int, default=256,
        help="Hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--d-e", type=int, default=128,
        help="Embedding dimension (default: 128)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=3,
        help="Message-passing layers (default: 3)",
    )
    parser.add_argument(
        "--num-heads", type=int, default=4,
        help="Attention heads (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda or cpu (default: auto)",
    )

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()


if __name__ == "__main__":
    main()
