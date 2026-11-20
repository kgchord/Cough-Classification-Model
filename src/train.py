"""Train a small CNN on Coswara cough + deep-breathing mel-spectrograms.

    python src/train.py --epochs 5 --batch 32

Saves models/best_model.h5 (best validation AUC) and models/metrics.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

MODELS_DIR = os.path.join(utils.ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.h5")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def build_model(input_shape=(utils.N_MELS, utils.N_FRAMES, 1)):
    """3 x (Conv -> BN -> ReLU -> MaxPool), global pooling, dropout, sigmoid output."""
    L = tf.keras.layers
    model = tf.keras.Sequential([L.Input(shape=input_shape)], name="coughsure_cnn")
    for filters in (16, 32, 64):
        model.add(L.Conv2D(filters, 3, padding="same", use_bias=False))
        # momentum 0.9 (default 0.99): with only ~65 steps the running stats must converge fast,
        # otherwise inference-mode BN collapses predictions to one class.
        model.add(L.BatchNormalization(momentum=0.9))
        model.add(L.ReLU())
        model.add(L.MaxPooling2D(2))
    model.add(L.GlobalAveragePooling2D())
    model.add(L.Dropout(0.3))
    model.add(L.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model


def features_for(df_all, X_all, part):
    """Rows of the precomputed feature matrix belonging to one split's participants."""
    mask = df_all["id"].isin(set(part["id"])).values
    return X_all[mask], df_all["label"].values[mask].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience on val AUC")
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = utils.load_metadata()
    if df.empty:
        sys.exit("No labelled recordings found. Run  python src/download_data.py  first.")
    train_df, test_df = utils.split_by_participant(df, test_size=0.2, seed=args.seed)
    # Validation split (also by participant) for checkpointing, so the test set stays untouched.
    fit_df, val_df = utils.split_by_participant(train_df, test_size=0.15, seed=args.seed)
    print(f"Recordings: {len(df)} | participants train/val/test: "
          f"{fit_df.id.nunique()}/{val_df.id.nunique()}/{test_df.id.nunique()} | "
          f"positive rate: {df.groupby('id').label.first().mean():.2f}")

    print("Computing mel-spectrograms (only new files; cached in data/features.npz) ...")
    X_all, _, df_feat = utils.build_features(df)
    X_fit, y_fit = features_for(df_feat, X_all, fit_df)
    X_val, y_val = features_for(df_feat, X_all, val_df)
    X_test, y_test = features_for(df_feat, X_all, test_df)
    X_fit_bal, y_fit_bal = utils.oversample_minority(X_fit, y_fit, seed=args.seed)
    print(f"Train samples: {len(y_fit)} -> {len(y_fit_bal)} after oversampling | "
          f"val {len(y_val)} | test {len(y_test)}")

    model = build_model()
    model.summary()
    ckpt = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max",
                                              save_best_only=True, verbose=0)
    callbacks = [ckpt,
                 tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=args.patience),
                 tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                                                      patience=max(2, args.patience // 2), verbose=0)]
    hist = model.fit(X_fit_bal, y_fit_bal, validation_data=(X_val, y_val), epochs=args.epochs,
                     batch_size=args.batch, callbacks=callbacks, verbose=2)

    best = tf.keras.models.load_model(MODEL_PATH)
    p_test = best.predict(X_test, batch_size=64, verbose=0).ravel()
    p_train = best.predict(X_fit, batch_size=64, verbose=0).ravel()
    metrics = {
        "train_acc": float(accuracy_score(y_fit, p_train >= 0.5)),
        "test_acc": float(accuracy_score(y_test, p_test >= 0.5)),
        "test_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else None,
        "n_train": int(len(y_fit)), "n_test": int(len(y_test)),
        "test_positive_rate": float(y_test.mean()),
        "best_val_auc": float(max(hist.history["val_auc"])),
        "epochs_run": len(hist.history["loss"]),
        "n_participants": int(df["id"].nunique()),
        "preprocessing": utils.PREPROC_VERSION,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    auc = f"{metrics['test_auc']:.3f}" if metrics["test_auc"] is not None else "n/a"
    print(f"\nResult: train acc {metrics['train_acc']:.1%} | test acc {metrics['test_acc']:.1%} "
          f"| test ROC-AUC {auc}  (n_test={len(y_test)}, positive rate {y_test.mean():.0%})")
    print(f"Saved {os.path.relpath(MODEL_PATH, utils.ROOT)}")


if __name__ == "__main__":
    main()
