from pathlib import Path

import pytest

from paithon import RuntimeConfig, RuntimeEngine
from tests.support.files import PROJECT_ROOT, load_module_from_path


EXAMPLE_PATH = PROJECT_ROOT / "examples" / "ml_mutag_demo.py"


def load_example_module():
    return load_module_from_path(EXAMPLE_PATH)


def build_synthetic_graphs(torch, Data):
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2],
        ],
        dtype=torch.long,
    )
    graphs = []
    for label in (0, 1, 0, 1):
        x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float,
        )
        y = torch.tensor([label], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs


def test_scripted_mutag_provider_covers_demo_functions():
    module = load_example_module()
    requirements_text = (PROJECT_ROOT / "requirements-ml-cpu.txt").read_text(encoding="utf-8")

    assert {
        "configure_reproducibility",
        "resolve_device",
        "load_mutag_dataset",
        "describe_dataset",
        "split_dataset",
        "build_loaders",
        "build_model",
        "build_optimizer",
        "count_trainable_parameters",
        "summarize",
    }.issubset(module.SCRIPTED_IMPLEMENTATIONS)
    assert {"train_epoch", "evaluate"}.issubset(module.SCRIPTED_REPAIRS)
    assert "requirements-ml-cpu.txt" in module.ML_STACK_HINT
    assert "torch==2.8.0" in requirements_text
    assert "torch_geometric==2.7.0" in requirements_text


def test_scripted_mutag_training_methods_heal_and_track_metrics(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    module = load_example_module()
    provider = module.ScriptedMutagProvider()
    engine = RuntimeEngine(provider=provider, config=RuntimeConfig(cache_dir=tmp_path))

    configure_reproducibility = module.make_configure_reproducibility(engine)
    resolve_device = module.make_resolve_device(engine)
    build_model = module.make_build_model(engine)
    build_optimizer = module.make_build_optimizer(engine)
    count_trainable_parameters = module.make_count_trainable_parameters(engine)
    MutagRun = module.make_training_run_class(engine)

    configure_reproducibility(11)
    device = resolve_device()
    graphs = build_synthetic_graphs(torch, Data)
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    model = build_model(3, 8, 2, dropout=0.0).to(device)
    optimizer = build_optimizer(model, lr=0.01, weight_decay=0.0)

    assert count_trainable_parameters(model) > 0

    run = MutagRun(model, optimizer, device, loader, loader)
    loss = run.train_epoch(1)
    accuracy = run.evaluate()
    summary = run.summarize("SYNTH", 1, len(graphs), len(graphs))

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert 0.0 <= accuracy <= 1.0
    assert summary["dataset"] == "SYNTH"
    assert summary["epochs"] == 1
    assert summary["train_graphs"] == len(graphs)
    assert summary["test_graphs"] == len(graphs)
    assert summary["test_accuracy"] is not None
    assert len(run.epoch_losses) == 1
    assert len(run.notes) == 2
    assert run.notes[0].startswith("healed train_epoch")
    assert run.notes[1].startswith("healed evaluate")
    assert any(call[:2] == ("repair", "train_epoch") for call in provider.calls)
    assert any(call[:2] == ("repair", "evaluate") for call in provider.calls)
    assert any(call[:2] == ("implement", "summarize") for call in provider.calls)
