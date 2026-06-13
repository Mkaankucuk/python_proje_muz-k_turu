#!/usr/bin/env python3
"""Convert the trained Keras model from python_proje_muz-k_turu to TFLite.

Example:
  python3 tools/convert_project_model_to_tflite.py \
    --model /path/to/python_proje_muz-k_turu/models/best_rnn_lstm_model.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

import tensorflow as tf


LABELS = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def _remove_null_quantization_config(value):
    if isinstance(value, dict):
        if "quantization_config" in value and value["quantization_config"] is None:
            value.pop("quantization_config")
        for key in ("renorm", "renorm_clipping", "renorm_momentum"):
            value.pop(key, None)
        for child in value.values():
            _remove_null_quantization_config(child)
    elif isinstance(value, list):
        for child in value:
            _remove_null_quantization_config(child)


def _keras_model_path_for_current_keras(model_path: Path, temp_dir: Path) -> Path:
    patched_path = temp_dir / model_path.name
    with zipfile.ZipFile(model_path, "r") as source:
        config = json.loads(source.read("config.json"))
        _remove_null_quantization_config(config)

        with zipfile.ZipFile(patched_path, "w") as target:
            for item in source.infolist():
                if item.filename == "config.json":
                    target.writestr(item, json.dumps(config))
                else:
                    target.writestr(item, source.read(item.filename))
    return patched_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", default=Path("assets/models"), type=Path)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = _keras_model_path_for_current_keras(
            args.model,
            Path(temporary_directory),
        )
        model = tf.keras.models.load_model(model_path, compile=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "genre_classifier.tflite"
    labels_path = args.output / "labels.json"
    model_path.write_bytes(tflite_model)
    labels_path.write_text(json.dumps(LABELS, indent=2), encoding="utf-8")

    print(f"Wrote {model_path}")
    print(f"Wrote {labels_path}")


if __name__ == "__main__":
    main()
