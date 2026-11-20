"""CoughSure Flask app: upload a cough .wav, get a screening risk level with Delhi air-quality context.

    python app.py      # http://127.0.0.1:5000
"""
import csv
import os
import sys
import tempfile
import uuid
from datetime import datetime

from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import air_quality  # noqa: E402
import utils  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "best_model.h5")
RESULTS_DIR = os.path.join(ROOT, "static", "results")
LOG_PATH = os.path.join(ROOT, "data", "screenings.csv")
LOG_FIELDS = ["time", "probability", "model_level", "overall_level", "aqi", "aqi_category"]

# Untuned starting thresholds on the model's probability; adjust after validation.
THRESHOLDS = {"medium": 0.4, "high": 0.7}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("No trained model found. Run  python src/train.py  first.")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def risk_level(p):
    if p >= THRESHOLDS["high"]:
        return "High"
    if p >= THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


MESSAGES = {
    "High": "This recording has acoustic features the model associates with respiratory infection. "
            "High risk – please consult a doctor.",
    "Medium": "Some features in this recording resemble those of infected coughs. "
              "Monitor your symptoms and consult a doctor if the cough persists.",
    "Low": "Likely a healthy cough.",
}


def log_screening(row):
    new = not os.path.exists(LOG_PATH)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def recent_screenings(n=8):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return list(csv.DictReader(f))[-n:][::-1]


@app.route("/")
def index():
    return render_template("index.html", air=air_quality.get_delhi_air_quality(),
                           history=recent_screenings())


@app.route("/api/air")
def api_air():
    aq = air_quality.get_delhi_air_quality()
    return (jsonify(aq), 200) if aq else (jsonify({"error": "air-quality data unavailable"}), 503)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f = request.files.get("audio")
        if f is None or f.filename == "":
            raise ValueError("No file uploaded. Please choose a .wav recording.")
        if not f.filename.lower().endswith(".wav"):
            raise ValueError("Only .wav files are supported.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            f.save(tmp.name)
        try:
            mel = utils.preprocess_file(tmp.name)
        except Exception as e:
            raise ValueError(f"Could not read the audio file ({e}).")
        finally:
            os.unlink(tmp.name)

        model = get_model()
        p = float(model.predict(mel[None, ..., None], verbose=0)[0, 0])
        model_level = risk_level(p)

        aq = air_quality.get_delhi_air_quality()
        overall, raised = air_quality.combine_risk(model_level, aq)

        # Spectrogram with saliency overlay for the result page.
        from explain import saliency_map, top_bands
        sal = saliency_map(model, mel)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        img_name = f"{uuid.uuid4().hex}.png"
        utils.save_mel_png(mel, os.path.join(RESULTS_DIR, img_name), saliency=sal)

        log_screening({"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "probability": f"{p:.3f}",
                       "model_level": model_level, "overall_level": overall,
                       "aqi": aq["aqi"] if aq else "", "aqi_category": aq["category"] if aq else ""})

        return render_template("result.html", probability=p, model_level=model_level,
                               overall_level=overall, raised_by_pollution=raised,
                               message=MESSAGES[model_level], air=aq,
                               air_advice=air_quality.pollution_advice(aq),
                               bands=top_bands(sal), image=f"results/{img_name}")
    except ValueError as e:
        return render_template("error.html", error=str(e)), 400
    except FileNotFoundError as e:
        return render_template("error.html", error=str(e)), 503
    except Exception as e:  # anything unexpected
        app.logger.exception("prediction failed")
        return render_template("error.html", error=f"Unexpected error: {e}"), 500


@app.errorhandler(413)
def too_large(_):
    return render_template("error.html", error="File too large (max 10 MB)."), 413


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
