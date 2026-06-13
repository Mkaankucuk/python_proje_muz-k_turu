#!/usr/bin/env python3
"""Train a GTZAN genre classifier and export it as TensorFlow Lite.

Expected dataset layout:

  genres_original/
    blues/blues.00000.wav
    classical/classical.00000.wav
    ...

Usage:
  python3 tools/train_gtzan_model.py --dataset /path/to/genres_original
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

FRAME_SIZE = 2048
HOP_SIZE = 1024
BAND_COUNT = 32
FEATURE_LENGTH = 37


def extract_features(path: Path) -> np.ndarray:
    samples, sample_rate = librosa.load(path, sr=None, mono=True)
    if len(samples) < FRAME_SIZE:
        raise ValueError(f"{path} is too short")

    stft = np.abs(
        librosa.stft(
            samples,
            n_fft=FRAME_SIZE,
            hop_length=HOP_SIZE,
            win_length=FRAME_SIZE,
            window="hamming",
            center=False,
        )
    )
    stft = stft[1:, :]
    bands = np.array_split(stft, BAND_COUNT, axis=0)
    band_features = [float(np.mean(np.log1p(np.sum(band, axis=0)))) for band in bands]

    rms = float(np.sqrt(np.mean(samples * samples)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(samples, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=stft, sr=sample_rate)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sample_rate)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sample_rate, roll_percent=0.85)))

    features = np.array(
        band_features
        + [
            rms,
            zcr,
            centroid / sample_rate,
            bandwidth / sample_rate,
            rolloff / sample_rate,
        ],
        dtype=np.float32,
    )
    if features.shape[0] != FEATURE_LENGTH:
        raise ValueError(f"Expected {FEATURE_LENGTH} features, got {features.shape[0]}")
    return features


def load_dataset(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[int] = []

    for label_index, label in enumerate(LABELS):
        genre_dir = dataset_dir / label
        if not genre_dir.exists():
            raise FileNotFoundError(f"Missing genre directory: {genre_dir}")
        for wav_path in sorted(genre_dir.glob("*.wav")):
            try:
                x_values.append(extract_features(wav_path))
                y_values.append(label_index)
            except Exception as exc:
                print(f"Skipping {wav_path}: {exc}")

    if not x_values:
        raise RuntimeError("No WAV files were loaded.")
    return np.vstack(x_values), np.array(y_values, dtype=np.int64)


def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_LENGTH,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path, help="Path to GTZAN genres_original directory")
    parser.add_argument("--output", default=Path("assets/models"), type=Path)
    parser.add_argument("--epochs", default=120, type=int)
    args = parser.parse_args()

    x_values, y_values = load_dataset(args.dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x_values,
        y_values,
        test_size=0.2,
        random_state=42,
        stratify=y_values,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=32,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=18,
                restore_best_weights=True,
            )
        ],
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # Fold the standardization into the first dense layer so Flutter only sends raw features.
    first_dense = model.layers[0]
    weights, bias = first_dense.get_weights()
    scale = scaler.scale_.astype(np.float32)
    mean = scaler.mean_.astype(np.float32)
    first_dense.set_weights([weights / scale[:, None], bias - (mean / scale) @ weights])

    args.output.mkdir(parents=True, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    (args.output / "genre_classifier.tflite").write_bytes(tflite_model)
    (args.output / "labels.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")
    print(f"Wrote {args.output / 'genre_classifier.tflite'}")
    print(f"Wrote {args.output / 'labels.json'}")


if __name__ == "__main__":
    main()
