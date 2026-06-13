from pathlib import Path
import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Dense, Dropout, LSTM, MaxPooling2D, Reshape, SimpleRNN
from tensorflow.keras.optimizers import Adam


SEED = 42
EPOCHS = 120
BATCH_SIZE = 64

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
        if (candidate / "dataset" / "features_3.0_sec.json").exists():
            return candidate
    raise FileNotFoundError("dataset/features_3.0_sec.json bulunamadı.")


def build_simple_rnn(input_shape):
    model = Sequential(name="simple_rnn_low")
    model.add(SimpleRNN(128, activation="tanh", input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(GENRES), activation="softmax"))
    return model


def build_cnn_lstm(input_shape):
    model = Sequential(name="cnn_lstm_high")
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(input_shape[0], input_shape[1], 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Reshape((33, 3 * 64)))
    model.add(LSTM(128, activation="tanh"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
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


def save_confusion_matrix(y_true, y_pred, title: str, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(GENRES)), labels=GENRES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(GENRES)), labels=GENRES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

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

    input_shape = X_train.shape[1:]
    X_train_cnn = X_train[..., np.newaxis]
    X_val_cnn = X_val[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    trials = [
        {
            "label": "Düşük başarı modeli - SimpleRNN",
            "builder": build_simple_rnn,
            "train_X": X_train,
            "val_X": X_val,
            "test_X": X_test,
            "learning_rate": 0.0001,
        },
        {
            "label": "Yüksek başarı adayı - CNN + LSTM",
            "builder": build_cnn_lstm,
            "train_X": X_train_cnn,
            "val_X": X_val_cnn,
            "test_X": X_test_cnn,
            "learning_rate": 0.001,
        },
    ]

    results = []
    trained_models = {}

    for trial in trials:
        label = trial["label"]
        print("=" * 80)
        print(label)
        print("=" * 80)

        model = trial["builder"](input_shape)
        model.compile(
            optimizer=Adam(learning_rate=trial["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        checkpoint_path = models_dir / f"{model.name}_checkpoint.keras"
        callbacks = [
            EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
        ]

        history = model.fit(
            trial["train_X"],
            y_train,
            validation_data=(trial["val_X"], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )

        train_loss, train_accuracy = model.evaluate(trial["train_X"], y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(trial["val_X"], y_val, verbose=0)
        test_loss, test_accuracy = model.evaluate(trial["test_X"], y_test, verbose=0)

        y_pred = np.argmax(model.predict(trial["test_X"], verbose=0), axis=1)
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        safe_name = model.name
        save_history_plot(history, label, models_dir / f"{safe_name}_history.png")
        save_confusion_matrix(y_test, y_pred, f"{label} - Confusion Matrix", models_dir / f"{safe_name}_confusion_matrix.png")

        report = classification_report(y_test, y_pred, target_names=GENRES, zero_division=0)
        report_path = models_dir / f"{safe_name}_classification_report.txt"
        report_path.write_text(report, encoding="utf-8")

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
                "precision_macro": precision_macro,
                "precision_weighted": precision_weighted,
                "recall_macro": recall_macro,
                "recall_weighted": recall_weighted,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "epochs_run": len(history.history["loss"]),
            }
        )

        print(f"Training accuracy:    {train_accuracy:.4f}")
        print(f"Validation accuracy:  {val_accuracy:.4f}")
        print(f"Test accuracy:        {test_accuracy:.4f}")
        print(f"Loss:                 {train_loss:.4f}")
        print(f"Validation loss:      {val_loss:.4f}")
        print(f"Test loss:            {test_loss:.4f}")
        print(f"Precision macro:      {precision_macro:.4f}")
        print(f"Precision weighted:   {precision_weighted:.4f}")
        print(f"Recall macro:         {recall_macro:.4f}")
        print(f"Recall weighted:      {recall_weighted:.4f}")
        print(f"F1 macro:             {f1_macro:.4f}")
        print(f"F1 weighted:          {f1_weighted:.4f}")

    comparison_df = pd.DataFrame(results).sort_values("validation_accuracy", ascending=False)
    comparison_path = models_dir / "rnn_lstm_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model"]
    best_model = trained_models[best_model_name]
    best_model.save(models_dir / "best_rnn_lstm_model.keras")
    best_model.save(models_dir / "best_rnn_lstm_model.h5")

    print("\nRNN/LSTM model karşılaştırma tablosu:")
    print(comparison_df)
    print(f"\nEn iyi RNN/LSTM model: {best_model_name}")
    print(f"Validation accuracy: {best_row['validation_accuracy']:.4f}")
    print(f"Kaydedildi: {models_dir / 'best_rnn_lstm_model.keras'}")
    print(f"Uyumluluk için kaydedildi: {models_dir / 'best_rnn_lstm_model.h5'}")


if __name__ == "__main__":
    main()
