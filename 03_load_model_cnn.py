from pathlib import Path
import json
import math

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


SEED = 42
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


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "models" / "best_model.keras").exists() or (
            candidate / "dataset" / "features_3.0_sec.json"
        ).exists():
            return candidate
    raise FileNotFoundError("Proje kökü bulunamadı.")


def extract_mfcc_segments_for_prediction(
    file_path: Path,
    sample_rate: int = 22500,
    track_duration: int = 30,
    segment_duration: float = 3.0,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 13,
) -> np.ndarray:
    signal, sr = librosa.load(file_path, sr=sample_rate, duration=track_duration)
    samples_per_segment = int(sample_rate * segment_duration)
    expected_frames = math.ceil(samples_per_segment / hop_length)
    num_segments = max(1, math.ceil(len(signal) / samples_per_segment))

    mfcc_segments = []
    for segment_index in range(num_segments):
        start_sample = segment_index * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment = signal[start_sample:end_sample]

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

        mfcc_segments.append(mfcc)

    if not mfcc_segments:
        raise ValueError("Tahmin için geçerli ses segmenti çıkarılamadı.")

    return np.array(mfcc_segments, dtype=np.float32)[..., np.newaxis]


def predict_genre(model, file_path: Path):
    X_audio = extract_mfcc_segments_for_prediction(file_path)
    probabilities = model.predict(X_audio, verbose=0)
    mean_probabilities = probabilities.mean(axis=0)
    predicted_index = int(np.argmax(mean_probabilities))
    predicted_genre = GENRES[predicted_index]
    return predicted_genre, predicted_index, mean_probabilities


def main() -> None:
    project_root = find_project_root()
    features_path = project_root / "dataset" / "features_3.0_sec.json"
    model_path = project_root / "models" / "best_model.keras"
    if not model_path.exists():
        model_path = project_root / "models" / "best_model.h5"

    if not model_path.exists():
        raise FileNotFoundError("Önce 02_train_cnn.py çalıştırılmalı.")

    model = tf.keras.models.load_model(model_path)

    with features_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"], dtype=np.float32)
    y = np.array(data["genre_num"], dtype=np.int64)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    X_test = X_test[..., np.newaxis]
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix - Best CNN Model")
    plt.tight_layout()
    confusion_path = project_root / "models" / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=160)
    plt.close(fig)
    print(f"Confusion matrix kaydedildi: {confusion_path}")

    sample_audio_path = project_root / "dataset" / "genres_original" / "blues" / "blues.00000.wav"
    genre_name, genre_index, probabilities = predict_genre(model, sample_audio_path)

    print(f"\nDosya: {sample_audio_path}")
    print(f"Tahmin: {genre_name.upper()} (Sınıf ID: {genre_index})")
    print("Olasılıklar:")
    for genre, probability in zip(GENRES, probabilities):
        print(f"  {genre:10s}: {probability * 100:5.2f}%")


if __name__ == "__main__":
    main()
