from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TraceDataset:
    hidden: torch.Tensor
    layer: torch.Tensor
    targets: torch.Tensor
    prompt_index: torch.Tensor
    prompt_ids: list[str]
    num_layers: int
    num_experts: int

    def __len__(self) -> int:
        return self.hidden.shape[0]


def load_split(
    directory: Path,
    split: str,
    maximum_pairs_per_prompt: int | None = None,
) -> TraceDataset:
    hidden_parts = []
    layer_parts = []
    target_parts = []
    prompt_parts = []
    prompt_ids = []
    observed_num_layers = 0
    observed_num_experts = 0

    for path in sorted(directory.glob("*.pt")):
        trace = torch.load(path, map_location="cpu", weights_only=False)
        if trace["split"] != split:
            continue
        features = trace["features"]
        routes = trace["route_ids"]
        steps, layers, hidden_size = features.shape
        observed_num_layers = max(observed_num_layers, trace.get("num_layers", layers))
        observed_num_experts = max(
            observed_num_experts,
            trace.get("num_experts", int(routes.max().item()) + 1),
        )
        pair_count = steps * layers
        hidden = features.reshape(pair_count, hidden_size)
        targets = routes.reshape(pair_count, routes.shape[-1]).long()
        layer = torch.arange(layers).repeat(steps)
        if maximum_pairs_per_prompt is not None and pair_count > maximum_pairs_per_prompt:
            # Deterministic coverage of the entire trajectory.
            selection = torch.linspace(
                0, pair_count - 1, maximum_pairs_per_prompt
            ).round().long()
            hidden = hidden[selection]
            targets = targets[selection]
            layer = layer[selection]
        prompt_number = len(prompt_ids)
        prompt_ids.append(trace["id"])
        hidden_parts.append(hidden)
        target_parts.append(targets)
        layer_parts.append(layer)
        prompt_parts.append(torch.full((hidden.shape[0],), prompt_number))

    if not hidden_parts:
        raise ValueError(f"No {split!r} traces found in {directory}")
    return TraceDataset(
        hidden=torch.cat(hidden_parts),
        layer=torch.cat(layer_parts),
        targets=torch.cat(target_parts),
        prompt_index=torch.cat(prompt_parts),
        prompt_ids=prompt_ids,
        num_layers=observed_num_layers,
        num_experts=observed_num_experts,
    )


def multi_hot(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    target = torch.zeros(
        indices.shape[0], num_experts, dtype=torch.float32, device=indices.device
    )
    return target.scatter_(1, indices.long(), 1.0)


def recall_at_k(scores: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    predictions = scores.topk(k, dim=-1).indices
    matches = (predictions[:, :, None] == targets[:, None, :]).any(dim=-1)
    return matches.float().sum(dim=-1) / targets.shape[-1]
