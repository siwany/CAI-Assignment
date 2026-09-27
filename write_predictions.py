"""Write ATIS test-set predictions in the Gradescope JSONL format."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

from evaluate import load_eval_data, load_model


def write_predictions(
    output_path: str,
    model_type: str,
    checkpoint_dir: str,
) -> None:
    sentences, _ = load_eval_data("test")
    model = load_model(model_type, checkpoint_dir)
    vocab = getattr(model, "vocab", None)
    predictions = model.predict(sentences, vocab)

    with Path(output_path).open("w", encoding="utf-8") as output_file:
        for index, (tokens, slots) in enumerate(zip(sentences, predictions)):
            json.dump(
                {"index": index, "tokens": tokens, "slots": slots},
                output_file,
            )
            output_file.write("\n")

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--crf-checkpoint", default="checkpoints/crf_1.0"
    )
    parser.add_argument(
        "--bilstm-checkpoint", default="checkpoints/bilstm_seed7"
    )
    args = parser.parse_args()

    write_predictions(
        "crf_predictions.jsonl",
        "crf",
        args.crf_checkpoint,
    )
    write_predictions(
        "bilstm_predictions.jsonl",
        "bilstm",
        args.bilstm_checkpoint,
    )
