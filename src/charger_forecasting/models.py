"""Compact neural baselines for panel time-series regression."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class DLinearRegressor(nn.Module):
    """Decomposition-linear next-day regression.

    A centered moving average separates trend and seasonal components before
    independent linear projections, following the DLinear construction in
    Zeng et al. (AAAI 2023).
    """

    def __init__(
        self,
        context_length: int,
        n_features: int,
        calendar_dim: int = 4,
        moving_average: int = 7,
    ) -> None:
        super().__init__()
        if moving_average <= 0 or moving_average % 2 == 0:
            raise ValueError("moving_average must be a positive odd integer")
        self.moving_average = moving_average
        self.seasonal = nn.Linear(context_length, 1)
        self.trend = nn.Linear(context_length, 1)
        self.calendar = nn.Linear(calendar_dim, n_features)

    def forward(
        self,
        x: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        channels = x.transpose(1, 2)
        padding = self.moving_average // 2
        padded = torch.nn.functional.pad(
            channels,
            (padding, padding),
            mode="replicate",
        )
        trend = torch.nn.functional.avg_pool1d(
            padded,
            kernel_size=self.moving_average,
            stride=1,
        )
        seasonal = channels - trend
        temporal = (
            self.seasonal(seasonal) + self.trend(trend)
        ).squeeze(-1)
        return temporal + self.calendar(calendar)


class LSTMRegressor(nn.Module):
    """Shared recurrent baseline for either entity definition."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        calendar_dim: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size + calendar_dim, n_features)

    def forward(
        self,
        x: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        return self.head(torch.cat((hidden[-1], calendar), dim=-1))


class TCNRegressor(nn.Module):
    """Small dilated temporal convolutional regression baseline."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        calendar_dim: int = 4,
    ) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(
                n_features,
                hidden_size,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=3,
                dilation=2,
                padding=2,
            ),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_size + calendar_dim, n_features)

    def forward(
        self,
        x: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.temporal(x.transpose(1, 2)).mean(dim=-1)
        return self.head(torch.cat((encoded, calendar), dim=-1))


class ITransformerRegressor(nn.Module):
    """Faithful inverted-token Transformer for one-step panel regression.

    Every physical variable is a token. Its complete lookback trajectory is
    projected into the model dimension, and self-attention learns
    cross-variable dependence as proposed by Liu et al. (ICLR 2024).
    """

    def __init__(
        self,
        context_length: int,
        n_features: int,
        *,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
        calendar_dim: int = 4,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.context_projection = nn.Linear(context_length, d_model)
        self.variable_embedding = nn.Parameter(
            torch.empty(1, n_features, d_model)
        )
        nn.init.normal_(
            self.variable_embedding,
            mean=0.0,
            std=1.0 / math.sqrt(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.calendar = nn.Linear(calendar_dim, n_features)

    def forward(
        self,
        x: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.context_projection(x.transpose(1, 2))
        tokens = self.encoder(tokens + self.variable_embedding)
        return self.head(tokens).squeeze(-1) + self.calendar(calendar)


def correlation_adjacency(
    train_values: np.ndarray,
    *,
    feature_index: int = 1,
    top_k: int = 3,
) -> np.ndarray:
    """Build a symmetric normalized graph using training data only."""

    if train_values.ndim != 3:
        raise ValueError("train_values must have shape [day, node, feature]")
    n_nodes = train_values.shape[1]
    if n_nodes == 0:
        raise ValueError("graph requires at least one node")
    if not 0 <= feature_index < train_values.shape[2]:
        raise ValueError("feature_index out of range")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    series = np.asarray(
        train_values[:, :, feature_index],
        dtype=np.float64,
    )
    centered = series - series.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((n_nodes, n_nodes), dtype=np.float64),
        where=denominator > 0.0,
    )
    correlation = np.abs(correlation)
    np.fill_diagonal(correlation, 0.0)

    adjacency = np.zeros_like(correlation)
    neighbours = min(top_k, max(0, n_nodes - 1))
    for node in range(n_nodes):
        if neighbours:
            indices = np.argsort(correlation[node])[-neighbours:]
            adjacency[node, indices] = correlation[node, indices]
    adjacency = np.maximum(adjacency, adjacency.T)
    adjacency += np.eye(n_nodes, dtype=np.float64)
    degree = adjacency.sum(axis=1)
    inverse = np.zeros_like(degree)
    positive = degree > 0.0
    inverse[positive] = degree[positive] ** -0.5
    normalized = inverse[:, None] * adjacency * inverse[None, :]
    return normalized.astype(np.float32)


class GraphTemporalRegressor(nn.Module):
    """GRU temporal encoder followed by one normalized graph-convolution."""

    def __init__(
        self,
        n_features: int,
        adjacency: np.ndarray,
        *,
        hidden_size: int = 32,
        calendar_dim: int = 4,
    ) -> None:
        super().__init__()
        matrix = torch.as_tensor(adjacency, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("adjacency must be square")
        self.register_buffer("adjacency", matrix)
        self.temporal = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.self_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_projection = nn.Linear(hidden_size, hidden_size)
        self.calendar = nn.Linear(calendar_dim, hidden_size)
        self.head = nn.Linear(hidden_size, n_features)

    def forward(
        self,
        x: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        batch, lookback, nodes, features = x.shape
        if nodes != self.adjacency.shape[0]:
            raise ValueError("node count does not match adjacency")
        sequence = x.permute(0, 2, 1, 3).reshape(
            batch * nodes,
            lookback,
            features,
        )
        _, hidden = self.temporal(sequence)
        hidden = hidden[-1].reshape(batch, nodes, -1)
        neighbours = torch.einsum(
            "nm,bmh->bnh",
            self.adjacency,
            hidden,
        )
        calendar_term = self.calendar(calendar).unsqueeze(1)
        encoded = torch.nn.functional.gelu(
            self.self_projection(hidden)
            + self.graph_projection(neighbours)
            + calendar_term
        )
        return self.head(encoded)
