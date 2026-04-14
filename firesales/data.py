"""
Data structures and graph construction for the holdings graph.

The holdings graph G_t = (I_t, A_t, E_t) is bipartite:
- I_t: investor nodes
- A_t: asset nodes
- E_t: position edges (i, a, w_ia,t, q_ia,t)

Trade target: y_{ia,t+1} = log(q_{ia,t+1} / q_{ia,t}), clipped to [-5, 5].
"""

import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class HoldingsGraph:
    """A single quarter's holdings graph G_t.

    Attributes:
        investor_num_features: Tensor [num_investors, num_investor_numerical_features]
        investor_cat_features: Tensor [num_investors, num_investor_categorical_features] (LongTensor)
        asset_num_features: Tensor [num_assets, num_asset_numerical_features]
        asset_cat_features: Tensor [num_assets, num_asset_categorical_features] (LongTensor)
        edge_index_ia: Tensor [2, num_edges] - (investor_idx, asset_idx) pairs
        edge_weight_w: Tensor [num_edges] - market-value position w_{ia,t}
        edge_weight_q: Tensor [num_edges] - notional-quantity position q_{ia,t}
        trade_target: Optional Tensor [num_edges] - y_{ia,t+1} = log(q_{t+1}/q_t), clipped [-5,5]
        trade_occurred: Optional Tensor [num_edges] - 1[y_{ia,t+1} != 0] (bool/float)
        fsi: Optional float - Financial Stress Index zeta_{t+1}
        quarter: Optional str - quarter identifier (e.g. "2020Q1")
    """
    investor_num_features: torch.Tensor
    investor_cat_features: torch.Tensor
    asset_num_features: torch.Tensor
    asset_cat_features: torch.Tensor
    edge_index_ia: torch.Tensor
    edge_weight_w: torch.Tensor
    edge_weight_q: torch.Tensor
    trade_target: Optional[torch.Tensor] = None
    trade_occurred: Optional[torch.Tensor] = None
    fsi: Optional[float] = None
    quarter: Optional[str] = None

    @property
    def num_investors(self) -> int:
        return self.investor_num_features.shape[0]

    @property
    def num_assets(self) -> int:
        return self.asset_num_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index_ia.shape[1]

    def to(self, device: torch.device) -> "HoldingsGraph":
        return HoldingsGraph(
            investor_num_features=self.investor_num_features.to(device),
            investor_cat_features=self.investor_cat_features.to(device),
            asset_num_features=self.asset_num_features.to(device),
            asset_cat_features=self.asset_cat_features.to(device),
            edge_index_ia=self.edge_index_ia.to(device),
            edge_weight_w=self.edge_weight_w.to(device),
            edge_weight_q=self.edge_weight_q.to(device),
            trade_target=self.trade_target.to(device) if self.trade_target is not None else None,
            trade_occurred=self.trade_occurred.to(device) if self.trade_occurred is not None else None,
            fsi=self.fsi,
            quarter=self.quarter,
        )


# ---------------------------------------------------------------------------
# Feature schema: defines the categorical vocabulary sizes and numerical
# feature count expected by the model. Users should adapt these to match
# their actual data.
# ---------------------------------------------------------------------------

# Investor categorical features and their vocabulary sizes
INVESTOR_CAT_FEATURES = {
    "institution_type": 10,   # e.g. open-end mutual fund, ETF, separate account, pension, insurance
    "manager_style": 10,      # e.g. active, passive, strategy types
}

# Investor numerical features (order matters -- must match tensor columns)
INVESTOR_NUM_FEATURES = [
    "total_aum",
    "num_positions",
    "avg_position_size",
    "std_position_size",
]

# Asset categorical features and their vocabulary sizes
ASSET_CAT_FEATURES = {
    "asset_class": 10,       # e.g. common equity, corporate bond, sovereign bond, fund shares
    "currency": 30,          # denomination currency
    "bond_subclass": 15,     # sub-class for debt securities (0 = N/A for equities)
}

# Asset numerical features
ASSET_NUM_FEATURES = [
    "amount_outstanding",
    "num_holders",
    "avg_position_size",
    "std_position_size",
    "coupon",
]


def build_holdings_graph(
    investors_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    next_holdings_df: Optional[pd.DataFrame] = None,
    fsi: Optional[float] = None,
    quarter: Optional[str] = None,
) -> HoldingsGraph:
    """Build a HoldingsGraph from DataFrames.

    Args:
        investors_df: One row per investor. Must contain columns matching
            INVESTOR_NUM_FEATURES and INVESTOR_CAT_FEATURES keys.
            Index = investor_id.
        assets_df: One row per asset. Must contain columns matching
            ASSET_NUM_FEATURES and ASSET_CAT_FEATURES keys.
            Index = asset_id.
        holdings_df: Columns: investor_id, asset_id, market_value (w), quantity (q).
            Only continuing positions with q > 0.
        next_holdings_df: Same schema as holdings_df but for period t+1.
            Used to compute trade targets. If None, trade_target will be None.
        fsi: Financial Stress Index value zeta_{t+1}.
        quarter: Quarter identifier string.

    Returns:
        HoldingsGraph
    """
    # Build investor id -> index mapping
    inv_ids = investors_df.index.tolist()
    inv_id_to_idx = {iid: idx for idx, iid in enumerate(inv_ids)}

    # Build asset id -> index mapping
    asset_ids = assets_df.index.tolist()
    asset_id_to_idx = {aid: idx for idx, aid in enumerate(asset_ids)}

    # Investor features
    inv_num = torch.tensor(
        investors_df[INVESTOR_NUM_FEATURES].values, dtype=torch.float32
    )
    inv_cat = torch.tensor(
        investors_df[list(INVESTOR_CAT_FEATURES.keys())].values, dtype=torch.long
    )

    # Asset features
    asset_num = torch.tensor(
        assets_df[ASSET_NUM_FEATURES].values, dtype=torch.float32
    )
    asset_cat = torch.tensor(
        assets_df[list(ASSET_CAT_FEATURES.keys())].values, dtype=torch.long
    )

    # Edges
    inv_indices = torch.tensor(
        [inv_id_to_idx[iid] for iid in holdings_df["investor_id"]], dtype=torch.long
    )
    asset_indices = torch.tensor(
        [asset_id_to_idx[aid] for aid in holdings_df["asset_id"]], dtype=torch.long
    )
    edge_index_ia = torch.stack([inv_indices, asset_indices], dim=0)
    edge_w = torch.tensor(holdings_df["market_value"].values, dtype=torch.float32)
    edge_q = torch.tensor(holdings_df["quantity"].values, dtype=torch.float32)

    # Trade targets: y_{ia,t+1} = log(q_{t+1}/q_t), clipped to [-5, 5]
    trade_target = None
    trade_occurred = None
    if next_holdings_df is not None:
        next_q_map = {}
        for _, row in next_holdings_df.iterrows():
            next_q_map[(row["investor_id"], row["asset_id"])] = row["quantity"]

        targets = []
        occurred = []
        for _, row in holdings_df.iterrows():
            key = (row["investor_id"], row["asset_id"])
            q_t = row["quantity"]
            q_tp1 = next_q_map.get(key, 0.0)
            if q_tp1 > 0 and q_t > 0:
                y = np.log(q_tp1 / q_t)
                y = np.clip(y, -5.0, 5.0)
            else:
                # Position exited -- treat as large sell (clipped)
                y = -5.0
            targets.append(y)
            occurred.append(1.0 if abs(y) > 1e-9 else 0.0)

        trade_target = torch.tensor(targets, dtype=torch.float32)
        trade_occurred = torch.tensor(occurred, dtype=torch.float32)

    return HoldingsGraph(
        investor_num_features=inv_num,
        investor_cat_features=inv_cat,
        asset_num_features=asset_num,
        asset_cat_features=asset_cat,
        edge_index_ia=edge_index_ia,
        edge_weight_w=edge_w,
        edge_weight_q=edge_q,
        trade_target=trade_target,
        trade_occurred=trade_occurred,
        fsi=fsi,
        quarter=quarter,
    )
