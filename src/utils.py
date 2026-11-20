"""Data preparation for CoughSure: metadata filtering, participant split, audio -> mel-spectrogram."""
import glob
import os

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "coswara")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "features.npz")

SR = 16000
N_MELS = 128
N_FRAMES = 128          # 128 frames * 512 hop / 16 kHz ~= 4.1 s window
N_FFT = 1024
HOP = 512
TOP_DB = 30             # silence threshold for trimming
FMIN = 50               # Hz; drops mic rumble / handling noise the first model latched onto
PREPROC_VERSION = "v2-fmin50"  # bump whenever preprocessing changes, to invalidate feature caches

# Coswara recording names kept: the two cough types and deep breathing.
SOUND_FILES = {
    "cough-heavy": "cough",
    "cough-shallow": "cough",
    "breathing-deep": "breathing",
}


# ---------------------------------------------------------------- metadata
def label_from_status(status):
    """1 = COVID-positive (any severity), 0 = healthy, None = ambiguous (dropped)."""
    s = str(status)
    if s.startswith("positive"):
        return 1
    if s == "healthy":
        return 0
    return None


def load_metadata(meta_path=META_PATH, audio_dir=AUDIO_DIR):
    """One row per available recording: id, folder, sound, sound_type, path, label."""
    meta = pd.read_csv(meta_path, usecols=["id", "covid_status"])
    meta["label"] = meta["covid_status"].map(label_from_status)
    meta = meta.dropna(subset=["label"]).astype({"label": int})
    labels = dict(zip(meta["id"], meta["label"]))

    rows = []
    for pdir in glob.glob(os.path.join(audio_dir, "*", "*")):
        pid = os.path.basename(pdir)
        if pid not in labels:
            continue
        for name, kind in SOUND_FILES.items():
            path = os.path.join(pdir, name + ".wav")
            if os.path.isfile(path) and os.path.getsize(path) > 1000:  # skip empty uploads
                rows.append({"id": pid, "folder": os.path.basename(os.path.dirname(pdir)),
                             "sound": name, "sound_type": kind, "path": path, "label": labels[pid]})
    return pd.DataFrame(rows, columns=["id", "folder", "sound", "sound_type", "path", "label"])


def split_by_participant(df, test_size=0.2, seed=42):
    """80/20 split on participant IDs (stratified by label) so no person is in both sets."""
    people = df.groupby("id")["label"].first()
    stratify = people.values if people.value_counts().min() >= 2 else None
    train_ids, test_ids = train_test_split(people.index, test_size=test_size,
                                           random_state=seed, stratify=stratify)
    return df[df["id"].isin(train_ids)].reset_index(drop=True), df[df["id"].isin(test_ids)].reset_index(drop=True)


# ---------------------------------------------------------------- audio
def load_audio(source, sr=SR):
    """Load any audio file (path or file-like) as mono float32 at `sr`."""
    y, _ = librosa.load(source, sr=sr, mono=True)
    return y.astype(np.float32)


def audio_to_mel(y, sr=SR):
    """Trim silence, compute log-mel spectrogram, fix to (N_MELS, N_FRAMES), scale to [0, 1]."""
    if y.size == 0 or not np.any(np.abs(y) > 1e-6):
        raise ValueError("Audio is empty or silent.")
    y_trim, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if y_trim.size >= N_FFT:
        y = y_trim
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fixed width: centre-crop long clips, zero(=min)-pad short ones.
    t = mel_db.shape[1]
    if t >= N_FRAMES:
        start = (t - N_FRAMES) // 2
        mel_db = mel_db[:, start:start + N_FRAMES]
    else:
        mel_db = np.pad(mel_db, ((0, 0), (0, N_FRAMES - t)), constant_values=mel_db.min())

    lo, hi = mel_db.min(), mel_db.max()
    return ((mel_db - lo) / (hi - lo + 1e-8)).astype(np.float32)


def preprocess_file(source):
    """Path / uploaded file -> model-ready mel array of shape (N_MELS, N_FRAMES)."""
    return audio_to_mel(load_audio(source))


# ---------------------------------------------------------------- dataset
def _safe_preprocess(path):
    try:
        return preprocess_file(path)
    except Exception:
        return None  # corrupt / silent recordings are common in crowd-sourced data


def build_features(df, cache_path=FEATURES_PATH, n_jobs=-1):
    """Mel features for every row, cached by relative file path + PREPROC_VERSION.

    Only files missing from the cache are computed (in parallel). Unreadable files are dropped.
    Returns X (n,128,128,1), y, kept df.
    """
    from joblib import Parallel, delayed

    cache = {}
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        if str(d["version"]) == PREPROC_VERSION:
            cache = {p: (m if m.size else None) for p, m in zip(d["paths"], d["mels"])}

    rel = [os.path.relpath(p, AUDIO_DIR) for p in df["path"]]
    todo = [i for i, r in enumerate(rel) if r not in cache]
    if todo:
        mels = Parallel(n_jobs=n_jobs)(delayed(_safe_preprocess)(df["path"].iloc[i]) for i in todo)
        cache.update({rel[i]: m for i, m in zip(todo, mels)})
        if cache_path:
            paths = np.array(list(cache), dtype=object)
            stored = np.empty(len(paths), dtype=object)
            stored[:] = [cache[p] if cache[p] is not None else np.zeros(0, np.float32) for p in paths]
            np.savez(cache_path, version=PREPROC_VERSION, paths=paths, mels=stored)

    keep = [i for i, r in enumerate(rel) if cache[r] is not None]
    kept = df.iloc[keep].reset_index(drop=True)
    X = (np.stack([cache[rel[i]] for i in keep])[..., None] if keep
         else np.zeros((0, N_MELS, N_FRAMES, 1), np.float32))
    return X, kept["label"].values.astype(np.float32), kept


def oversample_minority(X, y, seed=42):
    """Randomly duplicate minority-class samples until classes are balanced (training set only)."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() == counts.max():
        return X, y
    minority = classes[np.argmin(counts)]
    idx_min = np.where(y == minority)[0]
    extra = rng.choice(idx_min, counts.max() - counts.min(), replace=True)
    idx = rng.permutation(np.concatenate([np.arange(len(y)), extra]))
    return X[idx], y[idx]


def save_mel_png(mel, out_path, saliency=None, title=None):
    """Render a mel-spectrogram (optionally with a saliency overlay) to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3), dpi=110)
    ax.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    if saliency is not None:
        ax.imshow(saliency, origin="lower", aspect="auto", cmap="jet", alpha=0.45)
    ax.set_xlabel("time frame")
    ax.set_ylabel("mel band")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
