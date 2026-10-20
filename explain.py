"""
Explainable-AI + multilingual helpers for the Heart Disease Risk Predictor.

* explain_prediction()  - model-agnostic, per-patient feature contributions.
  For every input we replace its value with values drawn from the training data and
  measure how the predicted probability changes. A positive number means "this input
  pushed the risk UP compared with a typical patient", negative means "pushed it DOWN".
* build_report()        - turns inputs + prediction + contributions into plain English.
* translate()           - translates that English text to Hindi / Marathi with the Sarvam AI
  Translate API (https://docs.sarvam.ai). The API key is read from the SARVAM_API_KEY
  environment variable or a git-ignored .env file - never hard-coded and never shown in the UI.
"""
import base64
import json
import os
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import pandas as pd

import train

ROOT = Path(__file__).resolve().parent
SARVAM_URL = "https://api.sarvam.ai/translate"
SARVAM_MODEL = "sarvam-translate:v1"

LANGUAGES = {"English": "en-IN", "हिन्दी (Hindi)": "hi-IN", "मराठी (Marathi)": "mr-IN"}

FEATURE_LABEL = {
    "Age": "Age", "RestingBP": "Resting blood pressure", "Cholesterol": "Cholesterol",
    "MaxHR": "Maximum heart rate", "Oldpeak": "ST depression (Oldpeak)", "Sex": "Sex",
    "ChestPainType": "Chest pain type", "FastingBS": "Fasting blood sugar",
    "RestingECG": "Resting ECG", "ExerciseAngina": "Exercise-induced angina",
    "ST_Slope": "ST slope",
}
VALUE_TEXT = {
    "Sex": {"M": "Male", "F": "Female"},
    "ChestPainType": {"ASY": "Asymptomatic", "NAP": "Non-anginal pain",
                      "ATA": "Atypical angina", "TA": "Typical angina"},
    "FastingBS": {"0": "not above 120 mg/dL", "1": "above 120 mg/dL"},
    "RestingECG": {"Normal": "Normal", "ST": "ST-T wave abnormality",
                   "LVH": "Left ventricular hypertrophy"},
    "ExerciseAngina": {"N": "No", "Y": "Yes"},
    "ST_Slope": {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"},
}
UNITS = {"Age": "years", "RestingBP": "mm Hg", "Cholesterol": "mg/dL", "MaxHR": "bpm", "Oldpeak": ""}


def format_value(feature, value):
    if feature in VALUE_TEXT:
        return VALUE_TEXT[feature].get(str(value), str(value))
    unit = UNITS.get(feature, "")
    return f"{value} {unit}".strip()


# --------------------------------------------------------------------------- explainability
def explain_prediction(model, row, n_background=150, seed=0):
    """Return (probability, DataFrame[feature, contribution]) sorted by |contribution|."""
    cols = train.NUMERIC + train.CATEGORICAL
    background = train.load_data()[cols].sample(n=n_background, replace=True, random_state=seed)
    x = pd.DataFrame([row])[cols]
    proba = float(model.predict_proba(x)[0, 1])

    out = []
    for col in cols:
        perturbed = background.copy()
        perturbed[col] = row[col]            # everybody gets this patient's value for `col`...
        with_val = model.predict_proba(perturbed)[:, 1].mean()
        without = model.predict_proba(background)[:, 1].mean()   # ...versus the typical patient
        out.append((col, float(with_val - without)))
    df = pd.DataFrame(out, columns=["feature", "contribution"])
    df = df.reindex(df.contribution.abs().sort_values(ascending=False).index).reset_index(drop=True)
    return proba, df


# --------------------------------------------------------------------------- English report
def band(proba):
    if proba >= 0.65:
        return "High risk of heart disease"
    if proba >= 0.35:
        return "Moderate or borderline risk of heart disease"
    return "Low risk of heart disease"


def build_report(row, proba, contrib, top_n=5):
    """Return an ordered dict of English text sections (each short enough to translate)."""
    inputs = "; ".join(f"{FEATURE_LABEL[c]}: {format_value(c, row[c])}"
                       for c in train.NUMERIC + train.CATEGORICAL)
    result = f"{band(proba)}. Estimated probability: {proba:.0%}."

    up, down = [], []
    for _, r in contrib.head(top_n).iterrows():
        f, c = r.feature, r.contribution
        if abs(c) < 0.005:
            continue
        line = (f"{FEATURE_LABEL[f]} ({format_value(f, row[f])}) "
                f"{'raised' if c > 0 else 'lowered'} the risk by about {abs(c) * 100:.0f} percentage points.")
        (up if c > 0 else down).append(line)

    return {
        "inputs_title": "Your inputs",
        "inputs": inputs,
        "result_title": "Result",
        "result": result,
        "up_title": "Factors that increased your risk",
        "up": up or ["No input noticeably increased the risk."],
        "down_title": "Factors that decreased your risk",
        "down": down or ["No input noticeably decreased the risk."],
        "disclaimer": "This is an educational demonstration and not a medical diagnosis. "
                      "Please consult a doctor.",
    }


# --------------------------------------------------------------------------- Sarvam AI
def get_api_key():
    key = os.environ.get("SARVAM_API_KEY", "").strip()
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("SARVAM_API_KEY"):
                key = line.split("=", 1)[1].strip().strip("\"'")
    return key


def _translate_one(text, target, api_key):
    payload = json.dumps({
        "input": text, "source_language_code": "en-IN", "target_language_code": target,
        "model": SARVAM_MODEL,
    }).encode()
    req = urllib.request.Request(SARVAM_URL, data=payload, method="POST", headers={
        "Content-Type": "application/json", "api-subscription-key": api_key})
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read())["translated_text"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Sarvam API error {e.code}: {detail}") from None


SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
TTS_MODEL = "bulbul:v3"


def speech_script(report):
    """Text that is read aloud: result, then the main factors (skips the raw inputs list)."""
    parts = [report["result"], report["up_title"] + ".", *report["up"],
             report["down_title"] + ".", *report["down"], report["disclaimer"]]
    return " ".join(parts)


def text_to_speech(text, target, api_key, speaker="priya"):
    """Natural-sounding speech from Sarvam AI (bulbul:v3). Returns WAV bytes.
    Long text is split into chunks under the API limit and the WAV audio is joined."""
    import io
    import wave
    chunks, cur = [], ""
    for sentence in text.replace("\n", " ").split(" "):
        if len(cur) + len(sentence) + 1 > 1500:
            chunks.append(cur)
            cur = ""
        cur = f"{cur} {sentence}".strip()
    if cur:
        chunks.append(cur)

    ctx = ssl.create_default_context(cafile=certifi.where())
    buf, params = io.BytesIO(), None
    out = wave.open(buf, "wb")
    for chunk in chunks:
        payload = json.dumps({"text": chunk, "target_language_code": target,
                              "speaker": speaker, "model": TTS_MODEL}).encode()
        req = urllib.request.Request(SARVAM_TTS_URL, data=payload, method="POST", headers={
            "Content-Type": "application/json", "api-subscription-key": api_key})
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                audio = base64.b64decode(json.loads(resp.read())["audios"][0])
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Sarvam TTS error {e.code}: {e.read().decode(errors='replace')[:300]}") from None
        with wave.open(io.BytesIO(audio), "rb") as w:
            if params is None:
                params = w.getparams()
                out.setparams(params)
            out.writeframes(w.readframes(w.getnframes()))
    out.close()
    return buf.getvalue()


def translate_report(report, target, api_key):
    """Translate every string in the report to `target` (e.g. 'hi-IN'). Returns same structure."""
    if target == "en-IN":
        return report
    flat = []                                  # (key, index, text)
    for k, v in report.items():
        if isinstance(v, list):
            flat += [(k, i, t) for i, t in enumerate(v)]
        else:
            flat.append((k, None, v))
    with ThreadPoolExecutor(max_workers=6) as pool:
        translated = list(pool.map(lambda item: _translate_one(item[2], target, api_key), flat))
    out = {k: ([None] * len(v) if isinstance(v, list) else None) for k, v in report.items()}
    for (k, i, _), t in zip(flat, translated):
        if i is None:
            out[k] = t
        else:
            out[k][i] = t
    return out
