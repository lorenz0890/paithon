import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paithon import OpenAIProvider, RuntimeConfig, RuntimeEngine
from paithon.provider import LLMProvider


ML_STACK_HINT = (
    "This demo requires PyTorch and PyTorch Geometric. "
    "Install the CPU-only stack with `python -m pip install -r requirements-ml-cpu.txt`, "
    "then rerun this script. PyG will download MUTAG on the first run."
)

SCRIPTED_IMPLEMENTATIONS = {
    "configure_reproducibility": (
        "def configure_reproducibility(seed=7):\n"
        "    import random\n"
        "    import torch\n"
        "    random.seed(seed)\n"
        "    torch.manual_seed(seed)\n"
        "    if torch.cuda.is_available():\n"
        "        torch.cuda.manual_seed_all(seed)\n"
        "    return seed\n"
    ),
    "resolve_device": (
        "def resolve_device():\n"
        "    import torch\n"
        "    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    ),
    "load_mutag_dataset": (
        "def load_mutag_dataset(root):\n"
        "    from torch_geometric.datasets import TUDataset\n"
        "    dataset = TUDataset(root=root, name='MUTAG').shuffle()\n"
        "    if len(dataset) == 0:\n"
        "        raise ValueError('MUTAG dataset is empty')\n"
        "    if getattr(dataset, 'num_features', 0) <= 0:\n"
        "        raise ValueError('MUTAG demo expects node features to be present')\n"
        "    return dataset\n"
    ),
    "describe_dataset": (
        "def describe_dataset(dataset):\n"
        "    total_nodes = 0\n"
        "    total_edges = 0\n"
        "    for graph in dataset:\n"
        "        total_nodes += int(graph.num_nodes)\n"
        "        total_edges += int(graph.num_edges)\n"
        "    return {\n"
        "        'graphs': len(dataset),\n"
        "        'num_features': int(dataset.num_features),\n"
        "        'num_classes': int(dataset.num_classes),\n"
        "        'avg_nodes': round(total_nodes / max(len(dataset), 1), 2),\n"
        "        'avg_edges': round(total_edges / max(len(dataset), 1), 2),\n"
        "    }\n"
    ),
    "split_dataset": (
        "def split_dataset(dataset, train_ratio=0.8, seed=7):\n"
        "    import torch\n"
        "    total = len(dataset)\n"
        "    if total < 2:\n"
        "        raise ValueError('Need at least two graphs to create train and test splits')\n"
        "    generator = torch.Generator().manual_seed(seed)\n"
        "    indices = torch.randperm(total, generator=generator).tolist()\n"
        "    train_size = max(1, int(total * train_ratio))\n"
        "    train_size = min(train_size, total - 1)\n"
        "    train_indices = indices[:train_size]\n"
        "    test_indices = indices[train_size:]\n"
        "    return dataset[train_indices], dataset[test_indices]\n"
    ),
    "build_loaders": (
        "def build_loaders(train_dataset, test_dataset, batch_size=32):\n"
        "    from torch_geometric.loader import DataLoader\n"
        "    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n"
        "    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\n"
        "    return train_loader, test_loader\n"
    ),
    "build_model": (
        "def build_model(input_dim, hidden_dim, num_classes, dropout=0.2):\n"
        "    import torch\n"
        "    import torch.nn.functional as F\n"
        "    from torch_geometric.nn import GCNConv, global_mean_pool\n"
        "\n"
        "    class MutagGCN(torch.nn.Module):\n"
        "        def __init__(self):\n"
        "            super().__init__()\n"
        "            self.conv1 = GCNConv(input_dim, hidden_dim)\n"
        "            self.conv2 = GCNConv(hidden_dim, hidden_dim)\n"
        "            self.dropout = float(dropout)\n"
        "            self.classifier = torch.nn.Linear(hidden_dim, num_classes)\n"
        "\n"
        "        def forward(self, x, edge_index, batch):\n"
        "            x = self.conv1(x, edge_index)\n"
        "            x = F.relu(x)\n"
        "            x = F.dropout(x, p=self.dropout, training=self.training)\n"
        "            x = self.conv2(x, edge_index)\n"
        "            x = F.relu(x)\n"
        "            x = global_mean_pool(x, batch)\n"
        "            x = F.dropout(x, p=self.dropout, training=self.training)\n"
        "            return self.classifier(x)\n"
        "\n"
        "    return MutagGCN()\n"
    ),
    "build_optimizer": (
        "def build_optimizer(model, lr=0.01, weight_decay=5e-4):\n"
        "    import torch\n"
        "    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)\n"
    ),
    "count_trainable_parameters": (
        "def count_trainable_parameters(model):\n"
        "    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)\n"
    ),
    "summarize": (
        "def summarize(self, dataset_name, epochs, train_size, test_size):\n"
        "    return {\n"
        "        'dataset': dataset_name,\n"
        "        'epochs': epochs,\n"
        "        'train_graphs': train_size,\n"
        "        'test_graphs': test_size,\n"
        "        'best_epoch': self.best_epoch,\n"
        "        'epoch_losses': [round(float(loss), 4) for loss in self.epoch_losses],\n"
        "        'test_accuracy': None if self.test_accuracy is None else round(float(self.test_accuracy), 4),\n"
        "        'notes': list(self.notes),\n"
        "    }\n"
    ),
}

SCRIPTED_REPAIRS = {
    "train_epoch": (
        "def train_epoch(self, epoch):\n"
        "    import torch\n"
        "    self.model.train()\n"
        "    criterion = torch.nn.CrossEntropyLoss()\n"
        "    total_loss = 0.0\n"
        "    total_graphs = 0\n"
        "    for batch in self.train_loader:\n"
        "        batch = batch.to(self.device)\n"
        "        self.optimizer.zero_grad()\n"
        "        logits = self.model(batch.x, batch.edge_index, batch.batch)\n"
        "        loss = criterion(logits, batch.y)\n"
        "        loss.backward()\n"
        "        self.optimizer.step()\n"
        "        graphs = getattr(batch, 'num_graphs', int(batch.y.size(0)))\n"
        "        total_loss += float(loss.item()) * graphs\n"
        "        total_graphs += graphs\n"
        "    average_loss = total_loss / max(total_graphs, 1)\n"
        "    self.epoch_losses.append(average_loss)\n"
        "    if self.best_epoch is None:\n"
        "        self.best_epoch = epoch\n"
        "    else:\n"
        "        current_best = min(self.epoch_losses)\n"
        "        if average_loss <= current_best:\n"
        "            self.best_epoch = epoch\n"
        "    self.notes.append('healed train_epoch for epoch {0}'.format(epoch))\n"
        "    return average_loss\n"
    ),
    "evaluate": (
        "def evaluate(self):\n"
        "    import torch\n"
        "    self.model.eval()\n"
        "    correct = 0\n"
        "    total = 0\n"
        "    with torch.no_grad():\n"
        "        for batch in self.test_loader:\n"
        "            batch = batch.to(self.device)\n"
        "            logits = self.model(batch.x, batch.edge_index, batch.batch)\n"
        "            predictions = logits.argmax(dim=1)\n"
        "            correct += int((predictions == batch.y).sum().item())\n"
        "            total += int(batch.y.size(0))\n"
        "    self.test_accuracy = correct / max(total, 1)\n"
        "    self.notes.append('healed evaluate accuracy={0:.4f}'.format(self.test_accuracy))\n"
        "    return self.test_accuracy\n"
    ),
}


def ensure_ml_stack() -> None:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        raise SystemExit("{0}\nOriginal import error: {1}".format(ML_STACK_HINT, exc))


class ScriptedMutagProvider(LLMProvider):
    def __init__(self):
        self.calls = []

    def implement_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("implement", name, model))
        print("[llm] implement {0} via {1}".format(name, model))
        try:
            return SCRIPTED_IMPLEMENTATIONS[name]
        except KeyError as exc:
            raise KeyError("No scripted implementation for {0}".format(name)) from exc

    def repair_function(self, request, model):
        name = request.snapshot.name
        self.calls.append(("repair", name, request.error_type, model))
        print("[llm] repair {0} after {1} via {2}".format(name, request.error_type, model))
        try:
            return SCRIPTED_REPAIRS[name]
        except KeyError as exc:
            raise KeyError("No scripted repair for {0}".format(name)) from exc


def build_engine(provider_name: str, cache_dir: Path):
    config = RuntimeConfig(cache_dir=cache_dir, max_heal_attempts=1)
    if provider_name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        provider = OpenAIProvider()
    else:
        provider = ScriptedMutagProvider()
    return RuntimeEngine(provider=provider, config=config), provider


def make_configure_reproducibility(engine: RuntimeEngine):
    @engine.runtime_implemented
    def configure_reproducibility(seed=7):
        """Seed Python and torch RNGs for repeatable MUTAG experiments without using blocked system imports or mutating process environment variables."""
        raise NotImplementedError

    return configure_reproducibility


def make_resolve_device(engine: RuntimeEngine):
    @engine.runtime_implemented
    def resolve_device():
        """Return a torch.device that prefers CUDA when available and otherwise uses CPU."""
        raise NotImplementedError

    return resolve_device


def make_load_mutag_dataset(engine: RuntimeEngine):
    @engine.runtime_implemented
    def load_mutag_dataset(root):
        """Load torch_geometric.datasets.TUDataset(name='MUTAG') from root, shuffle it, and return the dataset."""
        raise NotImplementedError

    return load_mutag_dataset


def make_describe_dataset(engine: RuntimeEngine):
    @engine.runtime_implemented
    def describe_dataset(dataset):
        """Return a compact dict with graph count, feature count, class count, and average nodes and edges for a graph dataset."""
        raise NotImplementedError

    return describe_dataset


def make_split_dataset(engine: RuntimeEngine):
    @engine.runtime_implemented
    def split_dataset(dataset, train_ratio=0.8, seed=7):
        """Split a graph dataset into train and test subsets with a deterministic torch-based shuffle and keep at least one graph in each split."""
        raise NotImplementedError

    return split_dataset


def make_build_loaders(engine: RuntimeEngine):
    @engine.runtime_implemented
    def build_loaders(train_dataset, test_dataset, batch_size=32):
        """Build torch_geometric.loader.DataLoader instances for train and test graph datasets."""
        raise NotImplementedError

    return build_loaders


def make_build_model(engine: RuntimeEngine):
    @engine.runtime_implemented
    def build_model(input_dim, hidden_dim, num_classes, dropout=0.2):
        """Build a small torch_geometric GCN graph-classification model with two GCNConv layers, global mean pooling, and a linear classifier."""
        raise NotImplementedError

    return build_model


def make_build_optimizer(engine: RuntimeEngine):
    @engine.runtime_implemented
    def build_optimizer(model, lr=0.01, weight_decay=5e-4):
        """Return a torch.optim.Adam optimizer configured for a graph-classification model."""
        raise NotImplementedError

    return build_optimizer


def make_count_trainable_parameters(engine: RuntimeEngine):
    @engine.runtime_implemented
    def count_trainable_parameters(model):
        """Return the number of trainable parameters in a torch.nn.Module."""
        raise NotImplementedError

    return count_trainable_parameters


def make_training_run_class(engine: RuntimeEngine):
    class MutagRun:
        __paithon_version__ = "2025-03-ml-demo"

        def __init__(self, model, optimizer, device, train_loader, test_loader):
            self.model = model
            self.optimizer = optimizer
            self.device = device
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.epoch_losses = []
            self.test_accuracy = None
            self.best_epoch = None
            self.notes = []

        @engine.self_healing(
            state_fields=["epoch_losses", "best_epoch", "notes"],
            rollback_on_failure=True,
            rollback_fields=["model", "optimizer", "epoch_losses", "best_epoch", "notes"],
        )
        def train_epoch(self, epoch):
            """Train self.model for one epoch over self.train_loader, update tracked metrics, and return the average loss."""
            import torch

            self.notes.append("starting epoch {0}".format(epoch))
            self.model.train()
            criterion = torch.nn.CrossEntropyLoss()
            total_loss = 0.0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grd()
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item()) * getattr(batch, "num_graphs", int(batch.y.size(0)))
            average_loss = total_loss / max(len(self.train_loader.dataset), 1)
            self.epoch_losses.append(average_loss)
            self.best_epoch = epoch
            return average_loss

        @engine.self_healing(
            state_fields=["test_accuracy", "notes"],
            rollback_on_failure=True,
            rollback_fields=["model", "test_accuracy", "notes"],
        )
        def evaluate(self):
            """Evaluate self.model on self.test_loader, store accuracy in self.test_accuracy, append a note, and return the accuracy."""
            import torch

            self.notes.append("evaluating test split")
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in self.test_loader:
                    batch = batch.to(self.device)
                    logits = self.model(batch.x, batch.edge_index, batch.batch)
                    predictions = logits.argmax(dim=1)
                    correct += int((predictions == batch.targets).sum().item())
                    total += int(batch.targets.size(0))
            self.test_accuracy = correct / max(total, 1)
            return self.test_accuracy

        @engine.runtime_implemented(state_fields=["epoch_losses", "test_accuracy", "best_epoch", "notes"])
        def summarize(self, dataset_name, epochs, train_size, test_size):
            """Return a compact JSON-serializable dict summarizing the MUTAG run and tracked metrics."""
            raise NotImplementedError

    return MutagRun


def run_demo(
    provider_name: str,
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    dataset_root: Path,
    cache_dir: Path,
) -> None:
    ensure_ml_stack()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    engine, provider = build_engine(provider_name, cache_dir)
    configure_reproducibility = make_configure_reproducibility(engine)
    resolve_device = make_resolve_device(engine)
    load_mutag_dataset = make_load_mutag_dataset(engine)
    describe_dataset = make_describe_dataset(engine)
    split_dataset = make_split_dataset(engine)
    build_loaders = make_build_loaders(engine)
    build_model = make_build_model(engine)
    build_optimizer = make_build_optimizer(engine)
    count_trainable_parameters = make_count_trainable_parameters(engine)
    MutagRun = make_training_run_class(engine)

    print("PAIthon ML demo: MUTAG graph classification")
    print("provider =", provider_name)
    print("cache_dir =", cache_dir)
    print("dataset_root =", dataset_root)
    print("epochs =", epochs)

    configure_reproducibility(seed)
    device = resolve_device()
    dataset = load_mutag_dataset(str(dataset_root))
    dataset_stats = describe_dataset(dataset)
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.8, seed=seed)
    train_loader, test_loader = build_loaders(train_dataset, test_dataset, batch_size=batch_size)
    model = build_model(dataset.num_features, hidden_dim, dataset.num_classes, dropout=dropout).to(device)
    optimizer = build_optimizer(model, lr=learning_rate, weight_decay=weight_decay)
    parameter_count = count_trainable_parameters(model)

    print("\n1. Runtime-generated experiment setup")
    print("device ->", device)
    print("dataset stats ->", json.dumps(dataset_stats, sort_keys=True))
    print("split sizes -> train={0}, test={1}".format(len(train_dataset), len(test_dataset)))
    print("model parameters ->", parameter_count)

    run = MutagRun(model, optimizer, device, train_loader, test_loader)

    print("\n2. Self-healing training loop")
    for epoch in range(1, epochs + 1):
        loss = run.train_epoch(epoch)
        print("epoch {0} loss -> {1:.4f}".format(epoch, loss))

    print("\n3. Self-healing evaluation")
    accuracy = run.evaluate()
    print("test accuracy -> {0:.4f}".format(accuracy))

    print("\n4. Runtime-generated summary")
    summary = run.summarize("MUTAG", epochs, len(train_dataset), len(test_dataset))
    print(json.dumps(summary, indent=2, sort_keys=True))

    if isinstance(provider, ScriptedMutagProvider):
        print("\nLLM calls:")
        for call in provider.calls:
            print(" ", call)

        print("\n5. Cache reuse across a fresh engine")
        cached_engine, cached_provider = build_engine(provider_name, cache_dir)
        cached_resolve_device = make_resolve_device(cached_engine)
        cached_build_model = make_build_model(cached_engine)
        cached_count_parameters = make_count_trainable_parameters(cached_engine)
        cached_device = cached_resolve_device()
        cached_model = cached_build_model(dataset.num_features, hidden_dim, dataset.num_classes, dropout=dropout).to(cached_device)
        print("cached_device ->", cached_device)
        print("cached_model_parameters ->", cached_count_parameters(cached_model))
        if cached_provider.calls:
            print("fresh engine LLM calls ->", cached_provider.calls)
        else:
            print("fresh engine LLM calls -> none; setup functions loaded from cache")


def main():
    parser = argparse.ArgumentParser(description="Train a small GNN on MUTAG with PAIthon decorators.")
    parser.add_argument(
        "--provider",
        choices=("scripted", "openai"),
        default="scripted",
        help="Use the deterministic scripted provider or the real OpenAI provider.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for train and test loaders.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden width for the GNN.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Adam weight decay.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / ".paithon_datasets",
        help="Directory used by torch_geometric to download and cache MUTAG.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / ".paithon_cache" / "ml_mutag_demo",
        help="Directory used by PAIthon to cache generated and healed code.",
    )
    args = parser.parse_args()
    run_demo(
        args.provider,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        dataset_root=args.dataset_root,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
