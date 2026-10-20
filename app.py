"""
Heart Disease Risk Prediction - Streamlit app.

Run:  streamlit run app.py
(The model is trained automatically on first launch if model/heart_model.joblib is missing.)

Educational project only - not a medical device.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import explain
import train

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {background: linear-gradient(120deg,#e11d48 0%,#be123c 50%,#7c3aed 100%);
       color:#fff; padding:1.6rem 2rem; border-radius:18px; margin-bottom:1.5rem;}
.hero h1 {color:#fff; margin:0; font-size:2rem;}
.hero p {color:#ffe4e6; margin:.3rem 0 0;}
.card {border-radius:16px; padding:1.4rem 1.6rem; color:#fff; margin-bottom:1rem;}
.card h2 {color:#fff; margin:0 0 .2rem; font-size:2.6rem;}
.card p {margin:0; font-size:1.05rem;}
.high {background:linear-gradient(135deg,#dc2626,#9f1239);}
.mid  {background:linear-gradient(135deg,#f59e0b,#c2410c);}
.low  {background:linear-gradient(135deg,#10b981,#047857);}
.bar {height:14px; border-radius:7px; background:rgba(255,255,255,.3); margin-top:.8rem;}
.bar > div {height:100%; border-radius:7px; background:#fff;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training the model for the first time (about 20 seconds)...")
def load_model():
    if not train.MODEL_PATH.exists():
        train.main()
    model = joblib.load(train.MODEL_PATH)
    metrics = json.loads(train.METRICS_PATH.read_text()) if train.METRICS_PATH.exists() else {}
    return model, metrics


def risk_flags(d):
    """Plain-language list of inputs that are outside commonly cited healthy ranges."""
    flags = []
    if d["Age"] >= 55: flags.append(f"Age {d['Age']} (risk rises after ~55)")
    if d["RestingBP"] >= 140: flags.append(f"High resting blood pressure ({d['RestingBP']} mm Hg)")
    if d["Cholesterol"] >= 240: flags.append(f"High cholesterol ({d['Cholesterol']} mg/dL)")
    if d["FastingBS"] == "1": flags.append("Fasting blood sugar > 120 mg/dL")
    if d["MaxHR"] < 220 - d["Age"] - 60: flags.append(f"Low max heart rate for age ({d['MaxHR']} bpm)")
    if d["ExerciseAngina"] == "Y": flags.append("Chest pain during exercise")
    if d["Oldpeak"] >= 1.5: flags.append(f"ST depression of {d['Oldpeak']} (Oldpeak)")
    if d["ST_Slope"] in ("Flat", "Down"): flags.append(f"{d['ST_Slope']} ST slope")
    if d["ChestPainType"] == "ASY": flags.append("Asymptomatic chest pain type (strongest predictor in this data)")
    if d["RestingECG"] != "Normal": flags.append(f"Abnormal resting ECG ({d['RestingECG']})")
    return flags


model, metrics = load_model()

with st.sidebar:
    st.header("🌐 Language / भाषा")
    lang_name = st.radio("Show the report in", list(explain.LANGUAGES), index=0)
    lang_code = explain.LANGUAGES[lang_name]
    api_key = explain.get_api_key()
    st.caption("Translation and voice are powered by Sarvam AI.")
    voice_on = st.toggle("🔊 Read the explanation aloud", value=True)


@st.cache_data(show_spinner=False)
def cached_speech(text, target, key):
    return explain.text_to_speech(text, target, key)


@st.cache_data(show_spinner=False)
def cached_translation(report_json, target, key):
    return explain.translate_report(json.loads(report_json), target, key)

st.markdown('<div class="hero"><h1>❤️ Heart Disease Risk Predictor</h1>'
            '<p>Enter the clinical details below. The model returns a probability, not just yes/no.</p></div>',
            unsafe_allow_html=True)

tab_predict, tab_model = st.tabs(["🔍 Predict", "📊 Model performance"])

with tab_predict:
    with st.form("inputs"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 100, 50)
            sex = st.selectbox("Sex", ["M", "F"], format_func={"M": "Male", "F": "Female"}.get)
            chest_pain = st.selectbox("Chest pain type", ["ASY", "NAP", "ATA", "TA"], format_func={
                "ASY": "ASY – Asymptomatic", "NAP": "NAP – Non-anginal pain",
                "ATA": "ATA – Atypical angina", "TA": "TA – Typical angina"}.get)
            fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL", ["0", "1"],
                                      format_func={"0": "No", "1": "Yes"}.get)
        with c2:
            resting_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
            cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 220)
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], format_func={
                "Normal": "Normal", "ST": "ST – ST-T wave abnormality",
                "LVH": "LVH – Left ventricular hypertrophy"}.get)
        with c3:
            max_hr = st.slider("Max heart rate achieved", 60, 220, 140)
            exercise_angina = st.selectbox("Exercise-induced angina", ["N", "Y"],
                                           format_func={"N": "No", "Y": "Yes"}.get)
            oldpeak = st.slider("Oldpeak (ST depression)", -2.5, 6.5, 1.0, 0.1)
            st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])
        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if submitted:
        st.session_state["row"] = {
            "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol, "MaxHR": max_hr,
            "Oldpeak": oldpeak, "Sex": sex, "ChestPainType": chest_pain, "FastingBS": fasting_bs,
            "RestingECG": resting_ecg, "ExerciseAngina": exercise_angina, "ST_Slope": st_slope}

    # The last result stays on screen when the language is switched in the sidebar.
    if "row" in st.session_state:
        row = st.session_state["row"]
        proba, contrib = explain.explain_prediction(model, row)
        report = explain.build_report(row, proba, contrib)

        shown, note = report, None
        if lang_code != "en-IN":
            if not api_key:
                note = "Translation is not configured on this server. Showing English."
            else:
                try:
                    with st.spinner("Translating with Sarvam AI..."):
                        shown = cached_translation(json.dumps(report), lang_code, api_key)
                except Exception as e:  # network / invalid key
                    note = f"Translation failed ({e}). Showing English."
        if note:
            st.warning(note)

        cls = "high" if proba >= 0.65 else "mid" if proba >= 0.35 else "low"

        st.markdown(f"#### 📝 {shown['inputs_title']}")
        st.info(shown["inputs"])

        left, right = st.columns([1, 1])
        with left:
            st.markdown(f'<div class="card {cls}"><p>{shown["result_title"]}</p><h2>{proba:.0%}</h2>'
                        f'<p><b>{shown["result"]}</b></p><div class="bar"><div style="width:{proba*100:.0f}%"></div>'
                        f'</div></div>', unsafe_allow_html=True)
            st.caption("Bands: below 35% low · 35–65% moderate · above 65% high.")
        with right:
            st.markdown(f"**🔺 {shown['up_title']}**")
            for line in shown["up"]:
                st.markdown(f"- ⚠️ {line}")
            st.markdown(f"**🔻 {shown['down_title']}**")
            for line in shown["down"]:
                st.markdown(f"- ✅ {line}")

        st.markdown("**Why this prediction? (effect of each input on the risk, in percentage points)**")
        chart = contrib.assign(feature=contrib.feature.map(explain.FEATURE_LABEL),
                               points=contrib.contribution * 100).set_index("feature")[["points"]]
        st.bar_chart(chart.iloc[::-1], horizontal=True)
        st.caption("Right of zero = pushes risk up, left = pushes risk down, compared with a typical "
                   "patient in the training data.")
        st.info(shown["disclaimer"], icon="ℹ️")

        if voice_on:
            if not api_key:
                st.caption("Voice is not configured on this server.")
            else:
                try:
                    with st.spinner("Generating voice..."):
                        audio = cached_speech(explain.speech_script(shown), lang_code, api_key)
                    st.markdown("**🔊 Listen to the explanation**")
                    st.audio(audio, format="audio/wav", autoplay=True)
                except Exception as e:
                    st.warning(f"Voice could not be generated ({e}).")

with tab_model:
    if not metrics:
        st.warning("No metrics found. Run `python train.py`.")
    else:
        st.subheader(f"Selected model: {metrics['model']}")
        st.caption(f"Trained on {metrics['n_rows']} patients (UCI: Cleveland, Hungary, Switzerland, VA). "
                   f"Scores below are on {metrics['n_test']} held-out test patients the model never saw.")
        m = st.columns(5)
        for col, key, name in zip(m, ["accuracy", "precision", "recall", "f1", "roc_auc"],
                                  ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]):
            col.metric(name, f"{metrics[key]:.1%}" if key != "roc_auc" else f"{metrics[key]:.3f}")

        a, b = st.columns(2)
        with a:
            st.markdown("**Confusion matrix (test set)**")
            cm = metrics["confusion_matrix"]
            st.table(pd.DataFrame(cm, index=["Actual: no disease", "Actual: disease"],
                                  columns=["Predicted: no disease", "Predicted: disease"]))
        with b:
            st.markdown("**Cross-validated ROC AUC by algorithm**")
            cv = pd.Series(metrics["cv_roc_auc"]).sort_values()
            st.bar_chart(cv, horizontal=True)