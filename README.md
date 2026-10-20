# ❤️ Heart Disease Risk Predictor

A Streamlit app that estimates the probability of heart disease from 11 clinical inputs.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py          # optional – the app trains automatically on first launch
streamlit run app.py
```

## Project structure

```
app.py                  Streamlit web app (Predict + Model performance tabs)
train.py                Downloads data, compares 5 algorithms, saves best pipeline
requirements.txt
data/heart.csv          918 patients (UCI: Cleveland, Hungary, Switzerland, VA) – auto-downloaded
model/heart_model.joblib  Preprocessing + model in one pipeline
model/metrics.json      Test-set metrics and cross-validation scores
HeartdiseaseFinal.ipynb Original exploration notebook
```

## What was fixed vs. the original version

| Problem | Fix |
|---|---|
| `astype(int)` truncated Oldpeak (1.5 → 1) during training | Decimals kept |
| Cholesterol / BP of 0 treated as real values | Treated as missing and imputed |
| App hand-built dummy columns that could drift from training | One saved sklearn `Pipeline` does all encoding |
| Only a 0/1 label was shown | Probability + low / moderate / high band + risk-factor list |
| Model picked from a single test split | 5-fold cross-validation picks the model |
| Pickles tied to scikit-learn 1.6.1 | Model is retrained locally with your installed version |

> Educational project only – not a medical diagnosis.
