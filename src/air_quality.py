"""Live Delhi air quality and how it modulates the screening result.

Source (default, no API key): Open-Meteo Air Quality API (CAMS model estimates, hourly).
Optional: set WAQI_TOKEN to use station readings from aqicn.org instead.

The model score itself is never changed. Pollution only (a) adds an exposure level shown
beside the result and (b) can raise the *overall* advice level from Low to Medium on
Very Poor / Severe days, because sustained PM2.5 exposure is itself a respiratory risk.
"""
import os
from datetime import datetime

import requests

DELHI = {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090}
OPEN_METEO = "https://air-quality-api.open-meteo.com/v1/air-quality"

# India National AQI (CPCB) breakpoints: (conc_lo, conc_hi, aqi_lo, aqi_hi), 24-h averages, µg/m³.
NAQI_BREAKPOINTS = {
    "pm2_5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300),
              (120, 250, 300, 400), (250, 500, 400, 500)],
    "pm10": [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300),
             (350, 430, 300, 400), (430, 600, 400, 500)],
}
NAQI_CATEGORIES = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"),
                   (400, "Very Poor"), (500, "Severe")]


def naqi_subindex(pollutant, conc):
    """Linear interpolation inside the CPCB breakpoint band (capped at 500)."""
    for c_lo, c_hi, a_lo, a_hi in NAQI_BREAKPOINTS[pollutant]:
        if conc <= c_hi:
            return a_lo + (a_hi - a_lo) * (conc - c_lo) / (c_hi - c_lo)
    return 500.0


def naqi_category(aqi):
    for upper, name in NAQI_CATEGORIES:
        if aqi <= upper:
            return name
    return "Severe"


def exposure_level(aqi):
    """Map India AQI to the app's Low/Medium/High exposure scale."""
    if aqi <= 100:
        return "Low"
    if aqi <= 300:
        return "Medium"
    return "High"


def _from_open_meteo(loc, timeout):
    params = {"latitude": loc["lat"], "longitude": loc["lon"], "timezone": "Asia/Kolkata",
              "current": "pm2_5,pm10,nitrogen_dioxide,ozone,us_aqi",
              "hourly": "pm2_5,pm10", "past_days": 1, "forecast_days": 1}
    d = requests.get(OPEN_METEO, params=params, timeout=timeout).json()
    cur, hourly = d["current"], d["hourly"]
    # 24-h means over observed hours only (hourly also contains forecast hours).
    past = [i for i, t in enumerate(hourly["time"]) if t <= cur["time"]][-24:]

    def mean(key):
        vals = [hourly[key][i] for i in past if hourly[key][i] is not None]
        return sum(vals) / len(vals) if vals else cur[key]

    return {"source": "Open-Meteo (CAMS model estimate)", "time": cur["time"],
            "pm2_5_now": cur["pm2_5"], "pm10_now": cur["pm10"],
            "no2_now": cur.get("nitrogen_dioxide"), "o3_now": cur.get("ozone"),
            "pm2_5_24h": mean("pm2_5"), "pm10_24h": mean("pm10")}


def _from_waqi(token, timeout):
    d = requests.get("https://api.waqi.info/feed/delhi/", params={"token": token}, timeout=timeout).json()
    if d.get("status") != "ok":
        raise RuntimeError(f"WAQI error: {d.get('data')}")
    iaqi = d["data"]["iaqi"]
    # WAQI reports US-EPA sub-indices, not concentrations; keep them for display and
    # fall back to their AQI directly for the exposure level.
    return {"source": f"WAQI station: {d['data']['city']['name']}", "time": d["data"]["time"]["s"],
            "waqi_aqi": d["data"]["aqi"], "pm2_5_now": iaqi.get("pm25", {}).get("v"),
            "pm10_now": iaqi.get("pm10", {}).get("v"), "no2_now": iaqi.get("no2", {}).get("v"),
            "o3_now": iaqi.get("o3", {}).get("v")}


def get_delhi_air_quality(timeout=8):
    """Return a dict with pollutant values, India AQI, category and exposure level, or None if offline."""
    try:
        token = os.environ.get("WAQI_TOKEN")
        aq = _from_waqi(token, timeout) if token else _from_open_meteo(DELHI, timeout)
        if "pm2_5_24h" in aq:
            aqi = max(naqi_subindex("pm2_5", aq["pm2_5_24h"]), naqi_subindex("pm10", aq["pm10_24h"]))
        else:
            aqi = float(aq["waqi_aqi"])  # US AQI scale; close enough for a 3-level exposure flag
        aq.update({"aqi": round(aqi), "category": naqi_category(aqi), "exposure": exposure_level(aqi),
                   "location": DELHI["name"], "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
        return aq
    except Exception:
        return None


LEVELS = ["Low", "Medium", "High"]


def combine_risk(model_level, aq):
    """Overall advice level: model level, raised Low->Medium when Delhi AQI is Very Poor/Severe."""
    if aq and aq["aqi"] > 300 and model_level == "Low":
        return "Medium", True
    return model_level, False


def pollution_advice(aq):
    if not aq:
        return "Live air-quality data is unavailable right now."
    cat = aq["category"]
    if cat in ("Good", "Satisfactory"):
        return "Air quality is acceptable today."
    if cat == "Moderate":
        return "People with asthma or lung disease may feel discomfort; limit long outdoor exertion."
    if cat == "Poor":
        return "Breathing discomfort is likely on prolonged exposure. Reduce outdoor activity; consider an N95 mask outdoors."
    return ("Air quality is hazardous. Stay indoors where possible, use an N95 mask outside, and run an air purifier. "
            "Pollution alone can cause cough, so a persistent cough (>2 weeks) still needs a doctor's check.")
