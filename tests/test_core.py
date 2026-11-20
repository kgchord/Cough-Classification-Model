import glob
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
import air_quality  # noqa: E402
import utils  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "best_model.h5")


def synthetic_cough(sr=48000, seconds=3.0):
    """Silence + noisy burst + silence, at a non-16k rate to exercise resampling."""
    rng = np.random.default_rng(0)
    y = np.zeros(int(sr * seconds), np.float32)
    burst = slice(int(sr * 1.0), int(sr * 1.6))
    y[burst] = rng.normal(0, 0.3, burst.stop - burst.start) * np.hanning(burst.stop - burst.start)
    return y, sr


def wav_bytes(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------- preprocessing
def test_preprocess_shape_and_range(tmp_path):
    y, sr = synthetic_cough()
    path = tmp_path / "c.wav"
    sf.write(path, np.stack([y, y], axis=1), sr)  # stereo, 48 kHz
    mel = utils.preprocess_file(str(path))
    assert mel.shape == (utils.N_MELS, utils.N_FRAMES)
    assert mel.dtype == np.float32
    assert 0.0 <= mel.min() and mel.max() <= 1.0


def test_load_audio_is_mono_16k(tmp_path):
    y, sr = synthetic_cough()
    path = tmp_path / "c.wav"
    sf.write(path, np.stack([y, y], axis=1), sr)
    out = utils.load_audio(str(path))
    assert out.ndim == 1 and abs(len(out) - 3 * utils.SR) < 10


def test_silent_audio_rejected():
    with pytest.raises(ValueError):
        utils.audio_to_mel(np.zeros(utils.SR, np.float32))


# ---------------------------------------------------------------- data prep
def test_split_has_no_participant_leakage():
    df = pd.DataFrame({"id": np.repeat([f"p{i}" for i in range(50)], 3),
                       "label": np.repeat([i % 2 for i in range(50)], 3)})
    tr, te = utils.split_by_participant(df, test_size=0.2)
    assert set(tr.id).isdisjoint(te.id)
    assert te.id.nunique() == 10


def test_oversample_balances_classes():
    X = np.zeros((10, 2))
    y = np.array([0] * 8 + [1] * 2, np.float32)
    Xb, yb = utils.oversample_minority(X, y)
    assert (yb == 0).sum() == (yb == 1).sum() == 8 and len(Xb) == 16


def test_status_labels():
    assert utils.label_from_status("positive_mild") == 1
    assert utils.label_from_status("healthy") == 0
    assert utils.label_from_status("recovered_full") is None


# ---------------------------------------------------------------- air quality
@pytest.mark.parametrize("conc,expected", [(30, 50), (60, 100), (90, 200), (250, 400)])
def test_naqi_pm25_breakpoints(conc, expected):
    assert air_quality.naqi_subindex("pm2_5", conc) == pytest.approx(expected)


def test_pollution_raises_low_to_medium_only():
    severe = {"aqi": 420}
    assert air_quality.combine_risk("Low", severe) == ("Medium", True)
    assert air_quality.combine_risk("High", severe) == ("High", False)
    assert air_quality.combine_risk("Low", {"aqi": 150}) == ("Low", False)
    assert air_quality.combine_risk("Low", None) == ("Low", False)


# ---------------------------------------------------------------- model
@pytest.fixture(scope="module")
def model():
    tf = pytest.importorskip("tensorflow")
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    from train import build_model
    return build_model()


def test_model_prediction_on_known_sample(model):
    """Uses a real Coswara cough if downloaded, else the synthetic one."""
    real = sorted(glob.glob(os.path.join(utils.AUDIO_DIR, "*", "*", "cough-heavy.wav")))
    mel = None
    for path in real[:20]:
        try:
            mel = utils.preprocess_file(path)
            break
        except Exception:
            continue
    if mel is None:
        mel = utils.audio_to_mel(utils.load_audio(wav_bytes(*synthetic_cough())))
    p = model.predict(mel[None, ..., None], verbose=0)
    assert p.shape == (1, 1)
    assert 0.0 <= float(p[0, 0]) <= 1.0


def test_saliency_shape(model):
    from explain import saliency_map, top_bands
    mel = utils.audio_to_mel(utils.load_audio(wav_bytes(*synthetic_cough())))
    sal = saliency_map(model, mel)
    assert sal.shape == mel.shape and len(top_bands(sal)) == 3


# ---------------------------------------------------------------- web app
@pytest.fixture
def client(monkeypatch):
    import app as webapp
    monkeypatch.setattr(webapp.air_quality, "get_delhi_air_quality", lambda: None)
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"CoughSure" in r.data


def test_predict_without_file_shows_error(client):
    r = client.post("/predict", data={})
    assert r.status_code == 400 and b"No file uploaded" in r.data


def test_predict_rejects_non_wav(client):
    r = client.post("/predict", data={"audio": (io.BytesIO(b"x"), "a.mp3")},
                    content_type="multipart/form-data")
    assert r.status_code == 400
