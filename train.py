"""Train a CNN that classifies face images as 'with mask' or 'without mask'.

Expected folder layout (use any labelled face-mask image set):
    dataset/with_mask/*.jpg
    dataset/without_mask/*.jpg

Usage:  python train.py [--data dataset] [--epochs 30] [--img-size 128]
Outputs (in results/): metrics.json, confusion_matrix.png, training_curve.png
Model is saved to mask_cnn.keras
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SEED = 42
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
CLASS_NAMES = ["without_mask", "with_mask"]  # label 0 / label 1 (the model outputs P(with_mask))


def list_images(data_dir):
    paths, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        folder = os.path.join(data_dir, name)
        if not os.path.isdir(folder):
            raise SystemExit(f"Missing folder: {folder}")
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(EXTS):
                paths.append(os.path.join(folder, f))
                labels.append(label)
    return np.array(paths), np.array(labels)


def make_dataset(paths, labels, img_size, batch_size, training):
    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        return img, tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def build_model(img_size):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size, img_size, 3)),
        tf.keras.layers.Rescaling(1 / 255),
        # Augmentation is only active during training
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Conv2D(16, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"), tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--model-out", default="mask_cnn.keras")
    args = ap.parse_args()
    tf.random.set_seed(SEED)
    os.makedirs("results", exist_ok=True)

    paths, labels = list_images(args.data)
    print(f"Found {len(paths)} images: {int((labels == 1).sum())} with mask, {int((labels == 0).sum())} without")

    # Stratified 70 / 15 / 15 split by file, so the test set is never seen in training.
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, labels, test_size=0.30, stratify=labels, random_state=SEED)
    p_va, p_te, y_va, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f"Train {len(p_tr)} | Validation {len(p_va)} | Test {len(p_te)}")

    train_ds = make_dataset(p_tr, y_tr, args.img_size, args.batch_size, True)
    val_ds = make_dataset(p_va, y_va, args.img_size, args.batch_size, False)
    test_ds = make_dataset(p_te, y_te, args.img_size, args.batch_size, False)

    model = build_model(args.img_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[stop], verbose=2)
    model.save(args.model_out)

    probs = model.predict(test_ds, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    report = classification_report(y_te, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, preds)
    print("\nTest set results")
    print(classification_report(y_te, preds, target_names=CLASS_NAMES, zero_division=0))

    with open("results/metrics.json", "w") as f:
        json.dump({"n_train": len(p_tr), "n_val": len(p_va), "n_test": len(p_te),
                   "test_accuracy": report["accuracy"], "report": report,
                   "confusion_matrix": cm.tolist(), "class_names": CLASS_NAMES}, f, indent=2)

    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], CLASS_NAMES, rotation=20); plt.yticks([0, 1], CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig("results/confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch"); plt.legend(); plt.title("Training curve"); plt.tight_layout()
    plt.savefig("results/training_curve.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
