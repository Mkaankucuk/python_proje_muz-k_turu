from pathlib import Path
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


SEED = 42
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

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
        if (candidate / "dataset" / "features_3.0_sec.json").exists() or (
            candidate / "dataset" / "genres_original"
        ).exists():
            return candidate
    raise FileNotFoundError("Proje kökü bulunamadı.")


def build_cnn_baseline(input_shape):
    model = Sequential(name="cnn_baseline")
    model.add(Conv2D(32, 3, activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(3, strides=(2, 2), padding="same"))
    model.add(Conv2D(64, 3, activation="relu"))
    model.add(MaxPooling2D(3, strides=(2, 2), padding="same"))
    model.add(Conv2D(64, 2, activation="relu"))
    model.add(MaxPooling2D(2, strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(len(GENRES), activation="softmax"))
    return model


def build_cnn_best_candidate(input_shape):
    model = Sequential(name="cnn_best_candidate")
    model.add(Conv2D(32, 3, activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(3, strides=(2, 2), padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(3, strides=(2, 2), padding="same"))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, 2, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, strides=(2, 2), padding="same"))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(Dense(len(GENRES), activation="softmax"))
    return model


def save_history_plot(history, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history.history["accuracy"], label="training accuracy")
    axes[0].plot(history.history["val_accuracy"], label="validation accuracy")
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history.history["loss"], label="training loss")
    axes[1].plot(history.history["val_loss"], label="validation loss")
    axes[1].set_title(f"{title} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    project_root = find_project_root()
    features_path = project_root / "dataset" / "features_3.0_sec.json"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        raise FileNotFoundError("Önce 01_preprocessing.py çalıştırılmalı.")

    with features_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"], dtype=np.float32)
    y = np.array(data["genre_num"], dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train
    )

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    input_shape = X_train.shape[1:]

    builders = {
        "Önceki model - CNN baseline": build_cnn_baseline,
        "En başarılı model - CNN batchnorm/dropout": build_cnn_best_candidate,
    }

    trained_models = {}
    results = []

    for label, builder in builders.items():
        print("=" * 80)
        print(label)
        print("=" * 80)

        model = builder(input_shape)
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        checkpoint_path = models_dir / f"{model.name}_checkpoint.keras"
        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2,
        )

        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        safe_name = model.name.replace(" ", "_")
        save_history_plot(history, label, models_dir / f"{safe_name}_history.png")

        trained_models[label] = model
        results.append(
            {
                "model": label,
                "training_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "loss": train_loss,
                "validation_loss": val_loss,
                "test_loss": test_loss,
                "epochs_run": len(history.history["loss"]),
            }
        )

        print(f"Training accuracy:   {train_accuracy:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Test accuracy:       {test_accuracy:.4f}")
        print(f"Loss:                {train_loss:.4f}")
        print(f"Validation loss:     {val_loss:.4f}")
        print(f"Test loss:           {test_loss:.4f}")

    comparison_df = pd.DataFrame(results).sort_values("validation_accuracy", ascending=False)
    comparison_path = models_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]

    keras_path = models_dir / "best_model.keras"
    h5_path = models_dir / "best_model.h5"
    best_model.save(keras_path)
    best_model.save(h5_path)

    print("\nModel karşılaştırma tablosu:")
    print(comparison_df)
    print(f"\nEn iyi model: {best_model_name}")
    print(f"Validation accuracy: {best_row['validation_accuracy']:.4f}")
    print(f"Kaydedildi: {keras_path}")
    print(f"Uyumluluk için kaydedildi: {h5_path}")


if __name__ == "__main__":
    main()
