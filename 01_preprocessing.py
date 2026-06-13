from pathlib import Path
import json
import math

import librosa
import numpy as np


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
        if (candidate / "dataset" / "genres_original").exists():
            return candidate
    raise FileNotFoundError("dataset/genres_original bulunamadı.")


def extract_mfccs(
    dataset_dir: Path,
    output_path: Path,
    sample_rate: int = 22500,
    track_duration: int = 30,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 13,
    num_segments: int = 10,
) -> dict:
    data = {
        "mapping": GENRES,
        "genre_name": [],
        "genre_num": [],
        "file": [],
        "segment": [],
        "mfcc": [],
    }

    samples_per_track = sample_rate * track_duration
    samples_per_segment = samples_per_track // num_segments
    expected_mfcc_vectors = math.ceil(samples_per_segment / hop_length)

    print("MFCC extraction başladı")
    print("=======================")

    for genre_index, genre in enumerate(GENRES):
        genre_dir = dataset_dir / genre
        wav_files = sorted(genre_dir.glob("*.wav"))
        print(f"{genre.title():10s}: {len(wav_files)} dosya")

        for file_path in wav_files:
            try:
                signal, sr = librosa.load(file_path, sr=sample_rate, duration=track_duration)

                for segment_index in range(num_segments):
                    start_sample = segment_index * samples_per_segment
                    end_sample = start_sample + samples_per_segment
                    segment = signal[start_sample:end_sample]

                    if len(segment) < samples_per_segment:
                        segment = np.pad(segment, (0, samples_per_segment - len(segment)))

                    mfcc = librosa.feature.mfcc(
                        y=segment,
                        sr=sr,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mfcc=n_mfcc,
                    ).T

                    if len(mfcc) == expected_mfcc_vectors:
                        data["genre_name"].append(genre)
                        data["genre_num"].append(genre_index)
                        data["file"].append(str(file_path.relative_to(dataset_dir)))
                        data["segment"].append(segment_index)
                        data["mfcc"].append(mfcc.tolist())
            except Exception as exc:
                print(f"Atlandı: {file_path.name} -> {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)

    print("=======================")
    print(f"Kaydedildi: {output_path}")
    print(f"Toplam segment: {len(data['mfcc'])}")
    return data


def main() -> None:
    project_root = find_project_root()
    dataset_dir = project_root / "dataset" / "genres_original"
    output_path = project_root / "dataset" / "features_3.0_sec.json"

    print(f"Proje kökü: {project_root}")
    print(f"Veri seti: {dataset_dir}")
    print(f"Çıktı: {output_path}")

    missing = [genre for genre in GENRES if not (dataset_dir / genre).exists()]
    if missing:
        raise FileNotFoundError(f"Eksik tür klasörleri: {missing}")

    extract_mfccs(dataset_dir=dataset_dir, output_path=output_path)


if __name__ == "__main__":
    main()
