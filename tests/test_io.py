from pathlib import Path

import torch

from specstream.io import atomic_json
from specstream.predictor import LayerwiseExpertPredictor
from specstream.traces import load_split


def test_atomic_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "value.json"
    atomic_json(output, {"b": 2, "a": 1})
    assert output.read_text() == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_load_split_aligns_anchor_features_with_future_routes(
    tmp_path: Path,
) -> None:
    features = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
    routes = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[6, 7], [8, 9], [10, 11]],
        ]
    )
    torch.save(
        {
            "id": "prompt",
            "source": "synthetic",
            "split": "train",
            "router_features": features,
            "route_ids": routes,
            "num_layers": 3,
            "num_experts": 12,
        },
        tmp_path / "prompt.pt",
    )

    data = load_split(
        tmp_path,
        "train",
        feature_key="router_features",
        target_horizon=1,
    )

    assert data.hidden.shape == (4, 4)
    assert torch.equal(data.hidden, features[:, :2].reshape(4, 4))
    assert torch.equal(data.targets, routes[:, 1:].reshape(4, 2))
    assert torch.equal(data.layer, torch.tensor([1, 2, 1, 2]))


def test_layer_embedding_width_is_backward_compatible() -> None:
    legacy = LayerwiseExpertPredictor(16, 3, 4, width=8)
    preregistered = LayerwiseExpertPredictor(
        16, 3, 4, width=8, layer_embedding_width=2
    )

    assert legacy.layer_embedding.weight.shape == (3, 8)
    assert preregistered.layer_embedding.weight.shape == (3, 2)
    assert preregistered.metadata()["layer_embedding_width"] == 2
