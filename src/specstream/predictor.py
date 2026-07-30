from __future__ import annotations

import torch
from torch import nn


class LayerwiseExpertPredictor(nn.Module):
    """A small post-hoc predictor shared by every MoE layer."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_experts: int,
        width: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.width = width
        self.normalization = nn.LayerNorm(hidden_size)
        self.hidden_projection = nn.Linear(hidden_size, width, bias=False)
        self.layer_embedding = nn.Embedding(num_layers, width)
        self.output = nn.Linear(width, num_experts)

    def forward(self, hidden: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        state = self.hidden_projection(self.normalization(hidden))
        state = torch.nn.functional.gelu(state + self.layer_embedding(layer))
        return self.output(state)

    def metadata(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "width": self.width,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }

