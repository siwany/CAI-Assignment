"""Course-provided skeleton for training Part A (CRF) and Part B (BiLSTM) models.

In this training script, CLI parsing, data loading, and model construction are wired up for you.
The only one thing left to implement is `train_loop()` — actual training loop,
    loss_fn = torch.nn.CrossEntropyLoss()
for both Part A CRF and Part B BiLSTM. Look for the `# TODO: implement` block below.

Usage:
    python train.py --model crf    --train-fraction 1.00 --checkpoint-dir checkpoints/crf_1.00
    python train.py --model bilstm --train-fraction 0.25 --checkpoint-dir checkpoints/bilstm_0.25 \
        --tensorboard-logdir runs/bilstm_0.25
"""

from __future__ import annotations
import argparse
import random
from collections import Counter
from typing import Any
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    parser = argparse.ArgumentParser(
        description="Train a CRF or BiLSTM slot-filling model on ATIS."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["crf", "bilstm"],
        required=True,
        help="Which model to train.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction of training set to use, e.g. 0.05 / 0.10 / 0.25 / 1.00.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory to save the trained model to.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="BiLSTM dropout probability (default: 0.0).",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="BiLSTM hidden size per direction (default: 128).",
    )

    parser.add_argument(
        "--tensorboard-logdir",
        type=str,
        default=None,
        help="TensorBoard log directory. Only used when --model BiLSTM; "
        "ignored for --model crf (CRF training doesn't need TensorBoard or a GPU).",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch — CPU + CUDA — for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(fraction: float, seed: int) -> tuple[Any, Any, Any, Any]:
    """Load the ATIS train/dev/test splits, subsampling the train split.

    Returns:
        (`train_data`, `dev_data`, `test_data`, `vocab`)

        `train_data`/`dev_data`/`test_data` are `list[data_loader.Example]`.
        `vocab` is the dict returned by `build_vocab()` (see below),
        augmented with `tagset_size` (via `data_loader.slot_label_set()`) —
        needed to construct `BiLSTMModel` in `build_model()`.
    """
    import data_loader

    all_splits = data_loader.load_atis()
    train_data = data_loader.subsample(
        all_splits["train"], fraction=fraction, seed=seed
    )

    dev_data = all_splits["dev"]
    test_data = all_splits["test"]

    train_sentences = [example.tokens for example in train_data]
    vocab = build_vocab(train_sentences)
    vocab["tagset_size"] = len(data_loader.slot_label_set(train_data))

    return train_data, dev_data, test_data, vocab


# Built here rather than in data_loader.py to avoid touching a
# teammate-owned file; if data_loader.py later grows its own vocab
# utility, this should be removed in favor of that to avoid two
# divergent implementations.
def build_vocab(train_sentences: list[list[str]]) -> dict[str, Any]:
    """Build a token vocabulary from tokenized training sentences.

    Fully implemented — this is infrastructure, not the student-fill part
    of the assignment.

    Reserves index 0 for "<PAD>" and index 1 for "<UNK>". Tokens are then
    ordered by descending frequency, ties broken alphabetically. This
    determinism is required: `token_to_id` must be identical across
    machines/Python versions/runs, or it risks breaking the ±0.3 F1
    reproducibility tolerance in the grading rubric. Do NOT rely on dict or
    set iteration order for this — the explicit `sorted()` key below is
    what makes it deterministic.

    Args:
        train_sentences: Tokenized training sentences.

    Returns:
        A dict with `token_to_id` (Dict[str, int]), `id_to_token`
        (Dict[int, str]), and `vocab_size` (int).
    """
    counts = Counter(token for sentence in train_sentences for token in sentence)
    ordered_tokens = sorted(counts, key=lambda token: (-counts[token], token))

    token_to_id = {"<PAD>": 0, "<UNK>": 1}
    for token in ordered_tokens:
        token_to_id[token] = len(token_to_id)
    id_to_token = {index: token for token, index in token_to_id.items()}

    return {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "vocab_size": len(token_to_id),
    }


def build_model(
    model_type: str,
    vocab_size: int | None = None,
    tagset_size: int | None = None,
    dropout: float = 0.0,
    hidden_dim: int = 128,
) -> Any:
    """Construct an untrained model instance for the given model type.

    `vocab_size`/`tagset_size` (from `load_data()`'s `vocab`) are required
    for `model_type == "bilstm"` and ignored for `model_type == "crf"` —
    `CRFModel` works on raw token features, not a fixed vocabulary, so
    there's nothing for it to size itself against. This is intentional,
    not a bug, same as `CRFModel.predict()`'s unused `vocab` parameter.
    """
    if model_type == "crf":
        from models.crf import CRFModel

        return CRFModel()

    if model_type == "bilstm":
        from models.bilstm import BiLSTMModel

        if vocab_size is None or tagset_size is None:
            raise ValueError(
                "vocab_size and tagset_size are required for model_type='bilstm'"
            )

        return BiLSTMModel(
            vocab_size=vocab_size,
            tagset_size=tagset_size,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )

    raise ValueError(f"Unknown model type: {model_type!r}")


def train_loop(
    model: Any,
    train_data: Any,
    dev_data: Any,
    args: argparse.Namespace,
) -> float:
    """Train `model` on `train_data`, validating against `dev_data`.

    This is the part you implement.

    Your implementation must:
      - Train `model` on `train_data`.

      - Use `dev_data` for validation (and early stopping, if you choose to implement it).

      - When `args.model == "BiLSTM"`, log training progress to
        TensorBoard via `args.tensorboard_logdir` (e.g. using
        `torch.utils.tensorboard.SummaryWriter`). Not applicable for `args.model == "crf"`.

      - Save the final trained model to `args.checkpoint_dir`. See
        `evaluate.py`'s `load_model()` docstring for the exact checkpoint
        format your saved model must match.

    Returns:
        Final dev-set slot F1 (float). `main()` will print what you return.
    """
    import os

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.model == "crf":
        import joblib

        train_sentences = [example.tokens for example in train_data]
        train_labels = [example.slots for example in train_data]
        dev_sentences = [example.tokens for example in dev_data]
        dev_labels = [example.slots for example in dev_data]

        model.fit(train_sentences, train_labels)
        joblib.dump(model.model, os.path.join(args.checkpoint_dir, "crf_model.pkl"))

        dev_predictions = model.predict(dev_sentences)
        from scorer import span_scores
        return span_scores(dev_labels, dev_predictions)["f1"]

    if args.model == "bilstm":
        import torch
        from types import SimpleNamespace
        from torch.utils.tensorboard import SummaryWriter
        from scorer import span_scores

        train_sentences = [example.tokens for example in train_data]
        train_labels = [example.slots for example in train_data]
        dev_sentences = [example.tokens for example in dev_data]
        dev_labels = [example.slots for example in dev_data]

        tag_names = sorted({tag for labels in train_labels for tag in labels})
        tag_to_id = {tag: index for index, tag in enumerate(tag_names)}
        id_to_tag = {index: tag for tag, index in tag_to_id.items()}
        vocab = SimpleNamespace(
            token_to_id=build_vocab(train_sentences)["token_to_id"],
            id_to_tag=id_to_tag,
            tag_to_id=tag_to_id,
        )
        model.vocab = vocab

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        epochs = 25
        batch_size = 32
        writer = (
            SummaryWriter(log_dir=args.tensorboard_logdir)
            if args.tensorboard_logdir
            else None
        )

        best_dev_f1 = -1.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            order = list(range(len(train_data)))
            random.shuffle(order)
            total_loss = 0.0

            for start in range(0, len(order), batch_size):
                batch_indices = order[start : start + batch_size]
                batch_sentences = [train_sentences[index] for index in batch_indices]
                batch_labels = [train_labels[index] for index in batch_indices]
                lengths = torch.tensor(
                    [len(sentence) for sentence in batch_sentences], dtype=torch.long
                )
                max_length = int(lengths.max().item())
                token_batch = torch.zeros(
                    (len(batch_sentences), max_length), dtype=torch.long
                )
                label_batch = torch.full(
                    (len(batch_labels), max_length), -100, dtype=torch.long
                )

                for row, (sentence, labels) in enumerate(
                    zip(batch_sentences, batch_labels)
                ):
                    token_batch[row, : len(sentence)] = torch.tensor(
                        [vocab.token_to_id.get(token, 1) for token in sentence],
                        dtype=torch.long,
                    )
                    label_batch[row, : len(labels)] = torch.tensor(
                        [tag_to_id[tag] for tag in labels], dtype=torch.long
                    )

                optimizer.zero_grad()
                logits = model(token_batch, lengths)
                loss = loss_fn(
                    logits.reshape(-1, len(tag_names)), label_batch.reshape(-1)
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            predictions = model.predict(dev_sentences, vocab)
            dev_f1 = span_scores(dev_labels, predictions)["f1"]
            batch_count = max(1, (len(order) + batch_size - 1) // batch_size)
            average_loss = total_loss / batch_count

            if writer is not None:
                writer.add_scalar("loss/train", average_loss, epoch + 1)
                writer.add_scalar("f1/dev_span", dev_f1, epoch + 1)

            print(
                f"Epoch {epoch + 1:02d}/{epochs} "
                f"loss={average_loss:.4f} dev_span_f1={dev_f1:.4f}"
            )

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if writer is not None:
            writer.close()

        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab_size": model.embedding.num_embeddings,
                "tagset_size": len(tag_names),
                "embedding_dim": model.embedding.embedding_dim,
                "hidden_dim": model.lstm.hidden_size,
                "dropout": model.dropout.p,
                "vocab": {
                    "token_to_id": vocab.token_to_id,
                    "id_to_tag": vocab.id_to_tag,
                },
            },
            os.path.join(args.checkpoint_dir, "bilstm_model.pt"),
        )
        return best_dev_f1

    raise ValueError(f"Unknown model type: {args.model!r}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_data, dev_data, test_data, vocab = load_data(args.train_fraction, args.seed)

    if args.model == "bilstm":
        model = build_model(
            args.model,
            vocab_size=vocab["vocab_size"],
            tagset_size=vocab["tagset_size"],
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
        )

    else:
        model = build_model(args.model)

    dev_score = train_loop(model, train_data, dev_data, args)

    import os

    if not os.path.isdir(args.checkpoint_dir) or not os.listdir(args.checkpoint_dir):
        raise RuntimeError(
            f"Expected train_loop() to save a checkpoint to {args.checkpoint_dir!r}, "
            "but the directory is missing or empty."
        )

    print("Training complete.")
    print(f"  Model:          {args.model}")
    print(f"  Train fraction: {args.train_fraction}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    if dev_score is not None:
        print(f"  Final dev F1:   {dev_score:.4f}")


if __name__ == "__main__":
    main()
