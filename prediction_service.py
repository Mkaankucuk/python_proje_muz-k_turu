from functools import lru_cache
from pathlib import Path
import math

import librosa
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

GENRES = [
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

MODEL_OPTIONS = {
    "cnn": {
        "label": "CNN",
        "detail": "Validation accuracy: 0.7005 | Test accuracy: 0.6723",
        "path": MODELS_DIR / "best_model.keras",
        "feature_engineering": False,
    },
    "cnn_lstm": {
        "label": "CNN + LSTM",
        "detail": "Validation accuracy: 0.7863 | Test accuracy: 0.7684",
        "path": MODELS_DIR / "best_rnn_lstm_model.keras",
        "feature_engineering": False,
    },
    "fe_densenet": {
        "label": "DenseNet + Feature Engineering",
        "detail": "Validation accuracy: 0.7748 | Test accuracy: 0.7618",
        "path": MODELS_DIR / "feature_engineered_densenet.keras",
        "feature_engineering": True,
    },
    "fe_vit": {
        "label": "Vision Transformer + Feature Engineering",
        "detail": "Validation accuracy: 0.5833 | Test accuracy: 0.5542",
        "path": MODELS_DIR / "feature_engineered_vit.keras",
        "feature_engineering": True,
    },
}


@lru_cache(maxsize=None)
def load_model(model_key: str):
    model_path = MODEL_OPTIONS[model_key]["path"]
    if not model_path.exists():
        h5_fallback = model_path.with_suffix(".h5")
        if not h5_fallback.exists():
            raise FileNotFoundError(f"Model bulunamadı: {model_path.name}")
        model_path = h5_fallback
    return tf.keras.models.load_model(model_path)


def extract_mfcc_segments(
    file_path: Path,
    sample_rate: int = 22500,
    track_duration: int = 30,
    segment_duration: float = 3.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 13,
) -> np.ndarray:
    signal, sr = librosa.load(file_path, sr=sample_rate, duration=track_duration)
    if signal.size == 0:
        raise ValueError("Ses dosyası okunamadı.")

    samples_per_segment = int(sample_rate * segment_duration)
    expected_frames = math.ceil(samples_per_segment / hop_length)
    num_segments = max(1, math.ceil(len(signal) / samples_per_segment))

    segments = []
    for segment_index in range(num_segments):
        start = segment_index * samples_per_segment
        end = start + samples_per_segment
        segment = signal[start:end]
        if len(segment) == 0:
            continue
        if len(segment) < samples_per_segment:
            segment = np.pad(segment, (0, samples_per_segment - len(segment)))

        mfcc = librosa.feature.mfcc(
            y=segment,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mfcc=n_mfcc,
        ).T

        if mfcc.shape[0] < expected_frames:
            mfcc = np.pad(mfcc, ((0, expected_frames - mfcc.shape[0]), (0, 0)))
        elif mfcc.shape[0] > expected_frames:
            mfcc = mfcc[:expected_frames, :]
        segments.append(mfcc)

    if not segments:
        raise ValueError("Tahmin için geçerli ses segmenti çıkarılamadı.")
    return np.array(segments, dtype=np.float32)[..., np.newaxis]


@lru_cache(maxsize=1)
def load_feature_engineering_normalization():
    normalization_path = MODELS_DIR / "densenet_vit_feature_engineering_normalization.npz"
    if not normalization_path.exists():
        raise FileNotFoundError("Feature engineering normalizasyon dosyası bulunamadı.")
    values = np.load(normalization_path)
    return values["mean"], values["std"]


def apply_feature_engineering(mfcc_segments: np.ndarray) -> np.ndarray:
    raw_mfcc = mfcc_segments[..., 0]
    delta = librosa.feature.delta(raw_mfcc, axis=1)
    delta2 = librosa.feature.delta(raw_mfcc, order=2, axis=1)
    engineered = np.stack([raw_mfcc, delta, delta2], axis=-1).astype(np.float32)
    mean, std = load_feature_engineering_normalization()
    return (engineered - mean) / std


def predict_audio_file(file_path: Path, model_key: str) -> dict:
    model = load_model(model_key)
    audio_segments = extract_mfcc_segments(file_path)
    if MODEL_OPTIONS[model_key]["feature_engineering"]:
        audio_segments = apply_feature_engineering(audio_segments)
    predictions = model.predict(audio_segments, verbose=0)
    mean_probabilities = predictions.mean(axis=0)

    order = np.argsort(mean_probabilities)[::-1]
    best_index = int(order[0])

    probabilities = [
        {
            "genre": GENRES[int(index)],
            "percent": round(float(mean_probabilities[index]) * 100, 2),
        }
        for index in order
    ]

    return {
        "genre": GENRES[best_index],
        "confidence": probabilities[0]["percent"],
        "model_label": MODEL_OPTIONS[model_key]["label"],
        "segment_count": int(len(audio_segments)),
        "probabilities": probabilities,
    }
