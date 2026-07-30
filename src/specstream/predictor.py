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
        architecture: str = "layer_aware",
    ) -> None:
        super().__init__()
        if architecture not in {"layer_aware", "low_rank"}:
            raise ValueError(f"Unknown predictor architecture: {architecture}")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.width = width
        self.architecture = architecture
        self.normalization = (
            nn.LayerNorm(hidden_size) if architecture == "layer_aware" else nn.Identity()
        )
        self.hidden_projection = nn.Linear(hidden_size, width, bias=False)
        self.layer_embedding = (
            nn.Embedding(num_layers, width)
            if architecture == "layer_aware"
            else None
        )
        self.output = nn.Linear(
            width, num_experts, bias=architecture == "layer_aware"
        )

    def forward(self, hidden: torch.Tensor, layer: torch.Tensor) -> torch.Tensor:
        state = self.hidden_projection(self.normalization(hidden))
        if self.layer_embedding is not None:
            state = torch.nn.functional.gelu(state + self.layer_embedding(layer))
        return self.output(state)

    def metadata(self) -> dict[str, int | str]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "width": self.width,
            "architecture": self.architecture,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
        }
