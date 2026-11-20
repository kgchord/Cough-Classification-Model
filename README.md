# CoughSure – respiratory screening with Delhi air-quality context

Upload a cough `.wav` → a small CNN scores it → you get a Low / Medium / High risk banner, a
spectrogram with a saliency overlay, and today's **live Delhi air quality** (India AQI), which
can raise a Low result to Medium on Very Poor / Severe days. Every screening is logged locally
(`data/screenings.csv`) so the home page shows a running history alongside the AQI at the time.

> **Not a medical device.** The model is trained on **Coswara (COVID-19)** recordings, labelled
> COVID-positive vs healthy. It has **not** been trained or validated for tuberculosis. To make
> it a TB screener, retrain on a TB cough dataset (e.g. CODA TB DREAM Challenge) – the pipeline
> only needs a metadata CSV with a participant ID and a 0/1 label.

## Setup (Python 3.9+)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Get data
```bash
python src/download_data.py            # ~0.7 GB subset → data/coswara/, data/metadata.csv
python src/download_data.py --all --method git   # full dataset (~19 GB extracted), via git
```
`--method git` clones one folder at a time (no GitHub API rate limit), retries with back-off,
and moves a folder into place only once fully extracted. Rerun the same command to fill in
any folder that failed; finished folders are skipped.
Kept recordings: `cough-heavy`, `cough-shallow`, `breathing-deep`. Labels: `positive_*` → 1,
`healthy` → 0; other statuses (recovered, exposed, …) are dropped.

## 2. Train
```bash
python src/train.py --epochs 5 --batch 32
```
Participant-level 80/20 train/test split (plus a 15 % validation split from train for
checkpointing), minority-class oversampling on train only, 16 kHz mono, silence trimmed,
128×128 log-mel input (50 Hz–8 kHz). Features are cached in `data/features.npz` by file path,
so reruns only process new recordings. Early stopping / LR decay on validation AUC.
Writes `models/best_model.h5` and `models/metrics.json`.

**Current results** (`--epochs 30`, 42 of 43 folders – `20220224` failed to download):
1,904 participants, 5,695 recordings; test set 381 participants / 1,117 recordings.
Test ROC-AUC **0.74**, test accuracy 64% at a 0.5 threshold (below the 73% you'd get by always
predicting "healthy" – AUC is the meaningful number; thresholds need tuning). Averaged
saliency still peaks at 50–100 Hz, so some reliance on recording conditions can't be ruled out.
Metrics are per recording, not per participant.

## 3. Explain a recording
```bash
python src/explain.py data/coswara/<date>/<id>/cough-heavy.wav
```
Prints the top-3 mel bands by gradient saliency and writes `static/saliency_demo.png`.

## 4. Run the app
```bash
python app.py        # open http://127.0.0.1:5000
```
On macOS port 5000 is often taken by AirPlay – use `PORT=5001 python app.py`.
Upload any `cough-heavy.wav` from `data/coswara/…` (or `static/sample_cough.wav`). `GET /api/air` returns the live AQI as JSON.

Air quality comes from the free Open-Meteo API (CAMS model, no key). India AQI is computed
from 24-hour mean PM2.5/PM10 using CPCB breakpoints. For station readings, set
`WAQI_TOKEN=<your aqicn.org token>`.

## 5. Tests
```bash
pytest -q
```
Covers preprocessing (shape, range, resampling, silence), leakage-free split, oversampling,
AQI breakpoints, risk combination, model output and the Flask error paths.

## Layout
```
src/download_data.py  src/utils.py  src/train.py  src/explain.py  src/air_quality.py
app.py  templates/  static/  tests/  data/  models/
```
Risk thresholds (`app.py: THRESHOLDS`) are untuned defaults (0.4 / 0.7).

## Data & licence
Audio data: [Coswara](https://github.com/iiscleap/Coswara-Data) © 2021 LEAP Lab, Indian Institute
of Science, Bangalore, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`static/sample_cough.wav` is one unmodified Coswara recording, included for testing the app.
The dataset itself is not stored in this repo – download it with `src/download_data.py`.
