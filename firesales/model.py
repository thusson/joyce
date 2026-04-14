"""
Graph Transformer architecture for fire-sale prediction.

Implements the architecture from Clayton & Coppola (2026):
  - Bipartite holdings graph with investor and asset nodes
  - Feature embedding layer phi: x_v -> h^(0)_v (d_h-dimensional)
  - L layers of message passing with multi-head attention
  - Readout layer rho: h^(L) -> e_v (d_e-dimensional embeddings)
  - Masked autoencoder (MAE) head for self-supervised pretraining
  - Hurdle-model trade prediction head for supervised fine-tuning

All feed-forward layers use GELU non-linearities (Hendrycks & Gimpel, 2016).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .data import (
    HoldingsGraph,
    INVESTOR_CAT_FEATURES,
    INVESTOR_NUM_FEATURES,
    ASSET_CAT_FEATURES,
    ASSET_NUM_FEATURES,
)


# ============================================================================
# Configuration
# ============================================================================

class ModelConfig:
    """Hyperparameters from Table 2 of the paper."""

    def __init__(
        self,
        d_h: int = 256,          # Hidden dimension
        d_e: int = 128,          # Embedding (output) dimension
        num_layers: int = 3,     # Message-passing layers L
        num_heads: int = 4,      # Attention heads S_A
        d_c: int = 16,           # Categorical embedding dimension
        dropout: float = 0.1,    # Dropout rate
        # Feature schema sizes (must match data)
        investor_cat_vocab_sizes: Optional[dict] = None,
        asset_cat_vocab_sizes: Optional[dict] = None,
        num_investor_num_features: int = len(INVESTOR_NUM_FEATURES),
        num_asset_num_features: int = len(ASSET_NUM_FEATURES),
    ):
        self.d_h = d_h
        self.d_e = d_e
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_c = d_c
        self.dropout = dropout
        self.d_k = d_h // num_heads  # Per-head dimension d_k = d_v = d_h / S_A

        self.investor_cat_vocab_sizes = investor_cat_vocab_sizes or dict(INVESTOR_CAT_FEATURES)
        self.asset_cat_vocab_sizes = asset_cat_vocab_sizes or dict(ASSET_CAT_FEATURES)
        self.num_investor_num_features = num_investor_num_features
        self.num_asset_num_features = num_asset_num_features

        # Raw input dimensions (numerical + concatenated categorical embeddings)
        self.investor_input_dim = (
            num_investor_num_features
            + len(self.investor_cat_vocab_sizes) * d_c
        )
        self.asset_input_dim = (
            num_asset_num_features
            + len(self.asset_cat_vocab_sizes) * d_c
        )


# ============================================================================
# Building blocks
# ============================================================================

class FeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CategoricalEmbedder(nn.Module):
    """Embed each categorical feature into d_c dimensions, concatenate with numericals."""

    def __init__(self, vocab_sizes: dict, d_c: int):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, d_c)
            for name, vocab_size in vocab_sizes.items()
        })
        self.feature_names = list(vocab_sizes.keys())

    def forward(
        self, num_features: torch.Tensor, cat_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            num_features: [N, num_numerical]
            cat_features: [N, num_categorical] (LongTensor, column order matches feature_names)
        Returns:
            [N, num_numerical + num_categorical * d_c]
        """
        cat_embeds = []
        for i, name in enumerate(self.feature_names):
            cat_embeds.append(self.embeddings[name](cat_features[:, i]))
        cat_concat = torch.cat(cat_embeds, dim=-1)  # [N, num_cat * d_c]
        return torch.cat([num_features, cat_concat], dim=-1)


# ============================================================================
# Feature embedding: phi(x_v) -> h^(0)_v
# ============================================================================

class FeatureEmbedding(nn.Module):
    """Maps raw node characteristics x_v into initial d_h-dimensional embeddings.

    Equation (4): h^(0)_{v,t} = phi(x_{v,t})

    Separate embedding networks for investor and asset nodes, since they
    have different feature schemas.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.investor_cat_embedder = CategoricalEmbedder(
            config.investor_cat_vocab_sizes, config.d_c
        )
        self.asset_cat_embedder = CategoricalEmbedder(
            config.asset_cat_vocab_sizes, config.d_c
        )
        self.investor_proj = FeedForward(
            config.investor_input_dim, config.d_h, config.dropout
        )
        self.asset_proj = FeedForward(
            config.asset_input_dim, config.d_h, config.dropout
        )

    def forward(self, graph: HoldingsGraph):
        """Returns (investor_h0, asset_h0) each of shape [N, d_h]."""
        inv_raw = self.investor_cat_embedder(
            graph.investor_num_features, graph.investor_cat_features
        )
        inv_h0 = self.investor_proj(inv_raw)

        asset_raw = self.asset_cat_embedder(
            graph.asset_num_features, graph.asset_cat_features
        )
        asset_h0 = self.asset_proj(asset_raw)

        return inv_h0, asset_h0


# ============================================================================
# Message Passing Layer with Multi-Head Attention
# ============================================================================

class MessagePassingLayer(nn.Module):
    """One layer of bipartite message passing with attention.

    For each node v:
      1. Compute message: M^(l)_{v,t} = M^(l)(h^(l-1)_{v,t})   [Eq. 5]
      2. Compute normalized edge exposure: w_tilde_{vu,t}        [Eq. 6]
      3. Multi-head attention with log-bias from edge weights     [Eq. A.2]
      4. Aggregate messages: m^(l)_{v,t}                          [Eq. A.3]
      5. Update: h^(l)_{v,t} = U^(l)(h^(l-1)_{v,t}, m^(l)_{v,t}) [Eq. 7]

    Since the graph is bipartite, we pass messages in both directions:
      - Asset nodes aggregate messages from their investor neighbors
      - Investor nodes aggregate messages from their asset neighbors
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_h = config.d_h
        d_k = config.d_k
        num_heads = config.num_heads

        # Message function M^(l): R^d_h -> R^d_h  [Eq. 5]
        self.message_fn = FeedForward(d_h, d_h, config.dropout)

        # Per-head Q, K, V projection matrices [Eq. A.1]
        # Q, K: R^{d_k x d_h}, V: R^{d_k x d_h}
        self.W_q = nn.Linear(d_h, num_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_h, num_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_h, num_heads * d_k, bias=False)

        # Update function U^(l): R^{2*d_h} -> R^{d_h}  [Eq. 7]
        self.update_fn = FeedForward(2 * d_h, d_h, config.dropout)

        self.num_heads = num_heads
        self.d_k = d_k
        self.dropout = nn.Dropout(config.dropout)

    def _attend(
        self,
        h_target: torch.Tensor,
        h_source: torch.Tensor,
        edge_index: torch.Tensor,
        norm_weights: torch.Tensor,
        num_target: int,
    ) -> torch.Tensor:
        """Compute attention-weighted aggregated messages for one direction.

        Args:
            h_target: [N_target, d_h] embeddings of nodes receiving messages
            h_source: [N_source, d_h] embeddings of nodes sending messages
            edge_index: [2, E] - (target_idx, source_idx) pairs
            norm_weights: [E] - normalized edge exposure w_tilde_{vu,t}
            num_target: number of target nodes

        Returns:
            aggregated_messages: [N_target, d_h]
        """
        target_idx = edge_index[0]  # [E]
        source_idx = edge_index[1]  # [E]

        # Compute messages M^(l)(h^(l-1)_u) for source nodes
        M_source = self.message_fn(h_source)  # [N_source, d_h]

        # Q from target, K from source, V from source messages [Eq. A.1]
        Q = self.W_q(h_target)    # [N_target, num_heads * d_k]
        K = self.W_k(h_source)    # [N_source, num_heads * d_k]
        V = self.W_v(M_source)    # [N_source, num_heads * d_k]

        # Gather per-edge Q and K
        Q_edge = Q[target_idx]  # [E, num_heads * d_k]
        K_edge = K[source_idx]  # [E, num_heads * d_k]
        V_edge = V[source_idx]  # [E, num_heads * d_k]

        # Reshape for multi-head: [E, num_heads, d_k]
        E = target_idx.shape[0]
        Q_edge = Q_edge.view(E, self.num_heads, self.d_k)
        K_edge = K_edge.view(E, self.num_heads, self.d_k)
        V_edge = V_edge.view(E, self.num_heads, self.d_k)

        # Scaled dot-product attention scores with log-bias [Eq. A.2]
        # score = (Q^T K) / sqrt(d_k) + log(w_tilde)
        attn_scores = (Q_edge * K_edge).sum(dim=-1) / math.sqrt(self.d_k)  # [E, num_heads]

        # Add log-bias from normalized edge exposure
        log_bias = torch.log(norm_weights.clamp(min=1e-10)).unsqueeze(-1)  # [E, 1]
        attn_scores = attn_scores + log_bias  # [E, num_heads]

        # Neighborhood softmax: softmax over neighbors of each target node [Eq. A.2]
        attn_weights = _neighborhood_softmax(attn_scores, target_idx, num_target)  # [E, num_heads]
        attn_weights = self.dropout(attn_weights)

        # Weighted aggregation of values [Eq. A.3]
        weighted_V = attn_weights.unsqueeze(-1) * V_edge  # [E, num_heads, d_k]
        weighted_V = weighted_V.view(E, self.num_heads * self.d_k)  # [E, d_h]

        # Sum over neighbors (scatter_add) and average over heads [Eq. A.3]
        aggregated = torch.zeros(num_target, self.num_heads * self.d_k,
                                 device=h_target.device, dtype=h_target.dtype)
        aggregated.scatter_add_(0, target_idx.unsqueeze(-1).expand_as(weighted_V), weighted_V)

        return aggregated

    def forward(
        self,
        inv_h: torch.Tensor,
        asset_h: torch.Tensor,
        graph: HoldingsGraph,
        inv_norm_w: torch.Tensor,
        asset_norm_w: torch.Tensor,
    ):
        """One message-passing step on the bipartite graph.

        Args:
            inv_h: [I, d_h] investor embeddings from previous layer
            asset_h: [A, d_h] asset embeddings from previous layer
            graph: HoldingsGraph
            inv_norm_w: [E] normalized edge weight from investor perspective
            asset_norm_w: [E] normalized edge weight from asset perspective

        Returns:
            (inv_h_new, asset_h_new): updated embeddings
        """
        # Direction 1: investors aggregate from assets
        # edge_index for investor-as-target: (investor_idx, asset_idx)
        inv_msg = self._attend(
            h_target=inv_h,
            h_source=asset_h,
            edge_index=graph.edge_index_ia,
            norm_weights=inv_norm_w,
            num_target=graph.num_investors,
        )

        # Direction 2: assets aggregate from investors
        # Reverse edge_index: (asset_idx, investor_idx)
        edge_index_ai = torch.stack([graph.edge_index_ia[1], graph.edge_index_ia[0]], dim=0)
        asset_msg = self._attend(
            h_target=asset_h,
            h_source=inv_h,
            edge_index=edge_index_ai,
            norm_weights=asset_norm_w,
            num_target=graph.num_assets,
        )

        # Update: h^(l) = U^(l)(h^(l-1), m^(l))  [Eq. 7]
        inv_h_new = self.update_fn(torch.cat([inv_h, inv_msg], dim=-1))
        asset_h_new = self.update_fn(torch.cat([asset_h, asset_msg], dim=-1))

        return inv_h_new, asset_h_new


def _neighborhood_softmax(
    scores: torch.Tensor,
    target_idx: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    """Compute softmax of scores within each target node's neighborhood.

    Args:
        scores: [E, H] attention scores per edge per head
        target_idx: [E] which target node each edge belongs to
        num_targets: total number of target nodes

    Returns:
        [E, H] softmax-normalized attention weights
    """
    # For numerical stability, subtract max per neighborhood
    score_max = torch.zeros(num_targets, scores.shape[1],
                            device=scores.device, dtype=scores.dtype)
    score_max.scatter_reduce_(
        0,
        target_idx.unsqueeze(-1).expand_as(scores),
        scores,
        reduce="amax",
        include_self=True,
    )
    scores = scores - score_max[target_idx]

    exp_scores = scores.exp()

    # Sum of exp within each neighborhood
    sum_exp = torch.zeros(num_targets, scores.shape[1],
                          device=scores.device, dtype=scores.dtype)
    sum_exp.scatter_add_(0, target_idx.unsqueeze(-1).expand_as(exp_scores), exp_scores)

    return exp_scores / (sum_exp[target_idx] + 1e-10)


# ============================================================================
# GNN Backbone
# ============================================================================

class GNNBackbone(nn.Module):
    """The full GNN backbone: feature embedding + L message-passing layers + readout.

    Equations (4)-(8) from the paper.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.feature_embedding = FeatureEmbedding(config)
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(config) for _ in range(config.num_layers)
        ])
        # Readout layer rho: R^{d_h} -> R^{d_e}  [Eq. 8]
        self.readout = FeedForward(config.d_h, config.d_e, config.dropout)

    def _compute_normalized_edge_weights(self, graph: HoldingsGraph):
        """Compute w_tilde_{vu,t} per Eq. (6) for both directions.

        For investors: w_tilde_{ia} = w_{ia} / sum_{a' in N(i)} w_{ia'}
        For assets:    w_tilde_{ai} = w_{ai} / sum_{i' in N(a)} w_{ai'}

        Returns:
            (inv_norm_w, asset_norm_w): [E] each
        """
        w = graph.edge_weight_w  # [E]
        inv_idx = graph.edge_index_ia[0]  # [E]
        asset_idx = graph.edge_index_ia[1]  # [E]

        # Sum of weights per investor
        inv_sum = torch.zeros(graph.num_investors, device=w.device, dtype=w.dtype)
        inv_sum.scatter_add_(0, inv_idx, w)
        inv_norm_w = w / (inv_sum[inv_idx] + 1e-10)

        # Sum of weights per asset
        asset_sum = torch.zeros(graph.num_assets, device=w.device, dtype=w.dtype)
        asset_sum.scatter_add_(0, asset_idx, w)
        asset_norm_w = w / (asset_sum[asset_idx] + 1e-10)

        return inv_norm_w, asset_norm_w

    def forward(self, graph: HoldingsGraph):
        """Run the full GNN backbone.

        Returns:
            (investor_embeddings, asset_embeddings): [I, d_e] and [A, d_e]
        """
        inv_h, asset_h = self.feature_embedding(graph)
        inv_norm_w, asset_norm_w = self._compute_normalized_edge_weights(graph)

        for mp_layer in self.mp_layers:
            inv_h, asset_h = mp_layer(inv_h, asset_h, graph, inv_norm_w, asset_norm_w)

        # Readout [Eq. 8]
        inv_e = self.readout(inv_h)
        asset_e = self.readout(asset_h)

        return inv_e, asset_e


# ============================================================================
# Masked Autoencoder (MAE) Head
# ============================================================================

class MAEHead(nn.Module):
    """Masked autoencoder head for self-supervised pretraining.

    Predicts masked position sizes: w_hat_{ia} = f_AE(e_i, e_a)
    Loss: L_AE = sum_{masked} (g(w) - g(w_hat))^2
    where g(x) = sgn(x) * log(|x|)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(2 * config.d_e, config.d_h),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_h, 1),
        )

    @staticmethod
    def _g(x: torch.Tensor) -> torch.Tensor:
        """g(x) = sgn(x) * log(|x|)"""
        return x.sign() * (x.abs() + 1e-10).log()

    def forward(
        self,
        inv_e: torch.Tensor,
        asset_e: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict position sizes for given edges.

        Args:
            inv_e: [I, d_e]
            asset_e: [A, d_e]
            edge_index: [2, E_pred] edges to predict

        Returns:
            w_hat: [E_pred] predicted g(w) values
        """
        ei = inv_e[edge_index[0]]   # [E_pred, d_e]
        ea = asset_e[edge_index[1]]  # [E_pred, d_e]
        combined = torch.cat([ei, ea], dim=-1)  # [E_pred, 2*d_e]
        return self.predictor(combined).squeeze(-1)

    def loss(
        self,
        inv_e: torch.Tensor,
        asset_e: torch.Tensor,
        edge_index: torch.Tensor,
        true_w: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MAE loss on masked edges.

        L_AE = sum (g(w) - g(w_hat))^2
        """
        pred_gw = self.forward(inv_e, asset_e, edge_index)
        true_gw = self._g(true_w)
        return F.mse_loss(pred_gw, true_gw)


# ============================================================================
# Trade Prediction (Hurdle) Head
# ============================================================================

class TradePredictionHead(nn.Module):
    """Hurdle model trade prediction head.

    Input: concatenated (e_i, e_a, zeta_{t+1})
    Two branches:
      - Classification logits c_hat (extensive margin: does a trade occur?)
      - Regression prediction m_hat (intensive margin: how large?)
    Combined prediction [Eq. 9]:
      y_hat = sigma(c_hat) * m_hat

    Loss [Eq. A.4]:
      L_TP = sum BCE(c_hat, 1[y != 0]) + sum_{y != 0} (y - m_hat)^2
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        input_dim = 2 * config.d_e + 1  # e_i + e_a + zeta (scalar FSI)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.d_h),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Classification branch (extensive margin)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_h, config.d_h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_h // 2, 1),
        )

        # Regression branch (intensive margin)
        self.regressor = nn.Sequential(
            nn.Linear(config.d_h, config.d_h // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_h // 2, 1),
        )

    def forward(
        self,
        inv_e: torch.Tensor,
        asset_e: torch.Tensor,
        edge_index: torch.Tensor,
        fsi: float,
    ):
        """
        Args:
            inv_e: [I, d_e]
            asset_e: [A, d_e]
            edge_index: [2, E]
            fsi: scalar Financial Stress Index zeta_{t+1}

        Returns:
            c_hat: [E] classification logits
            m_hat: [E] regression predictions
            y_hat: [E] combined predictions = sigma(c_hat) * m_hat
        """
        ei = inv_e[edge_index[0]]   # [E, d_e]
        ea = asset_e[edge_index[1]]  # [E, d_e]

        # Broadcast FSI scalar to match edge count
        fsi_tensor = torch.full(
            (ei.shape[0], 1), fsi, device=ei.device, dtype=ei.dtype
        )
        combined = torch.cat([ei, ea, fsi_tensor], dim=-1)  # [E, 2*d_e + 1]

        trunk_out = self.trunk(combined)
        c_hat = self.classifier(trunk_out).squeeze(-1)  # [E]
        m_hat = self.regressor(trunk_out).squeeze(-1)    # [E]
        y_hat = torch.sigmoid(c_hat) * m_hat             # [E], Eq. 9

        return c_hat, m_hat, y_hat

    def loss(
        self,
        c_hat: torch.Tensor,
        m_hat: torch.Tensor,
        trade_target: torch.Tensor,
        trade_occurred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hurdle model loss [Eq. A.4].

        L_TP = BCE(c_hat, 1[y != 0]) + MSE(m_hat, y) for y != 0
        """
        # Extensive margin: binary cross-entropy
        bce_loss = F.binary_cross_entropy_with_logits(c_hat, trade_occurred)

        # Intensive margin: MSE on non-zero trades only
        nonzero_mask = trade_occurred > 0.5
        if nonzero_mask.any():
            mse_loss = F.mse_loss(m_hat[nonzero_mask], trade_target[nonzero_mask])
        else:
            mse_loss = torch.tensor(0.0, device=c_hat.device)

        return bce_loss + mse_loss


# ============================================================================
# Full Graph Transformer Model
# ============================================================================

class FireSaleGraphTransformer(nn.Module):
    """Complete graph transformer model.

    Combines:
      - GNN backbone (feature embedding + message passing + readout)
      - MAE head (self-supervised pretraining)
      - Trade prediction head (supervised fine-tuning)

    Training procedure (Section 3, Eq. A.5):
      Phase 1: min_{Theta_Backbone, Theta_AE} L_AE  (pretrain)
      Phase 2: min_{Theta_Backbone, Theta_TP} L_TP   (fine-tune, MAE head frozen)

    ~1.3M parameters total.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.backbone = GNNBackbone(self.config)
        self.mae_head = MAEHead(self.config)
        self.trade_head = TradePredictionHead(self.config)

    def forward_embeddings(self, graph: HoldingsGraph):
        """Run backbone to get embeddings.

        Returns:
            (inv_e, asset_e): investor and asset embeddings [I/A, d_e]
        """
        return self.backbone(graph)

    def forward_mae(self, graph: HoldingsGraph, mask_rate: float = 0.35):
        """Forward pass for MAE pretraining.

        Masks `mask_rate` fraction of edges, runs backbone on remaining edges,
        then predicts masked position sizes.

        Args:
            graph: HoldingsGraph
            mask_rate: fraction of edges to mask (default 0.35 per Table 2)

        Returns:
            mae_loss: scalar loss
        """
        num_edges = graph.num_edges
        num_mask = int(num_edges * mask_rate)

        # Random mask
        perm = torch.randperm(num_edges, device=graph.edge_index_ia.device)
        mask_idx = perm[:num_mask]
        keep_idx = perm[num_mask:]

        # Build masked graph (only keep edges)
        masked_graph = HoldingsGraph(
            investor_num_features=graph.investor_num_features,
            investor_cat_features=graph.investor_cat_features,
            asset_num_features=graph.asset_num_features,
            asset_cat_features=graph.asset_cat_features,
            edge_index_ia=graph.edge_index_ia[:, keep_idx],
            edge_weight_w=graph.edge_weight_w[keep_idx],
            edge_weight_q=graph.edge_weight_q[keep_idx],
        )

        # Get embeddings from masked graph
        inv_e, asset_e = self.backbone(masked_graph)

        # Predict masked positions
        masked_edge_index = graph.edge_index_ia[:, mask_idx]
        masked_true_w = graph.edge_weight_w[mask_idx]

        return self.mae_head.loss(inv_e, asset_e, masked_edge_index, masked_true_w)

    def forward_trade_prediction(self, graph: HoldingsGraph):
        """Forward pass for trade prediction.

        Args:
            graph: HoldingsGraph with trade_target, trade_occurred, and fsi set.

        Returns:
            (loss, c_hat, m_hat, y_hat)
        """
        inv_e, asset_e = self.backbone(graph)
        c_hat, m_hat, y_hat = self.trade_head(
            inv_e, asset_e, graph.edge_index_ia, graph.fsi
        )

        loss = self.trade_head.loss(
            c_hat, m_hat, graph.trade_target, graph.trade_occurred
        )

        return loss, c_hat, m_hat, y_hat

    def predict(self, graph: HoldingsGraph, fsi: float):
        """Generate trade predictions for a given graph and FSI scenario.

        Args:
            graph: HoldingsGraph (positions at time t)
            fsi: Financial Stress Index scenario value zeta_{t+1}

        Returns:
            y_hat: [E] predicted trade quantities y_hat = sigma(c_hat) * m_hat
            c_hat: [E] classification logits (extensive margin)
            m_hat: [E] regression predictions (intensive margin)
        """
        self.eval()
        with torch.no_grad():
            inv_e, asset_e = self.backbone(graph)
            c_hat, m_hat, y_hat = self.trade_head(
                inv_e, asset_e, graph.edge_index_ia, fsi
            )
        return y_hat, c_hat, m_hat

    def get_embeddings(self, graph: HoldingsGraph):
        """Extract learned investor and asset embeddings.

        These are the final representations e_{v,t} from the readout layer
        [Eq. 8], useful for downstream analysis such as:
        - Explaining cross-section of returns in stress episodes
        - Systemic risk measurement
        - Fire-sale vulnerability analysis

        Returns:
            (inv_e, asset_e): [I, d_e] and [A, d_e]
        """
        self.eval()
        with torch.no_grad():
            return self.backbone(graph)
