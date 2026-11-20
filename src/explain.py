"""Gradient saliency for the CoughSure CNN.

Interpretation: bright (red/yellow) regions in the overlay are the time-frequency
areas whose energy most changed the model's score for this recording.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils  # noqa: E402


def saliency_map(model, mel):
    """|d score / d input| for one mel array (N_MELS, N_FRAMES), smoothed and scaled to [0, 1]."""
    import tensorflow as tf
    from scipy.ndimage import gaussian_filter

    x = tf.convert_to_tensor(mel[None, ..., None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        score = model(x, training=False)[:, 0]
    grad = np.abs(tape.gradient(score, x).numpy()[0, ..., 0])
    grad = gaussian_filter(grad, sigma=2)
    return grad / (grad.max() + 1e-8)


def top_bands(saliency, k=3, sr=utils.SR):
    """Top-k mel bands by mean saliency, returned as [(band_idx, centre_hz, importance)]."""
    import librosa

    band_importance = saliency.mean(axis=1)
    freqs = librosa.mel_frequencies(n_mels=utils.N_MELS, fmin=utils.FMIN, fmax=sr / 2)
    order = np.argsort(band_importance)[::-1][:k]
    return [(int(b), float(freqs[b]), float(band_importance[b])) for b in order]


if __name__ == "__main__":
    # python src/explain.py path/to/cough.wav  -> prints top-3 bands, writes static/saliency_demo.png
    import tensorflow as tf

    model = tf.keras.models.load_model(os.path.join(utils.ROOT, "models", "best_model.h5"))
    mel = utils.preprocess_file(sys.argv[1])
    sal = saliency_map(model, mel)
    for b, hz, imp in top_bands(sal):
        print(f"mel band {b:3d} (~{hz:,.0f} Hz): importance {imp:.2f}")
    out = os.path.join(utils.ROOT, "static", "saliency_demo.png")
    utils.save_mel_png(mel, out, saliency=sal, title="Saliency (bright = most influential)")
    print("Wrote", os.path.relpath(out, utils.ROOT))
