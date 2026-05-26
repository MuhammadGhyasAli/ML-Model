import joblib
import pandas as pd
import numpy as np
import os
import functools
from flask import Flask, request, render_template, jsonify

app = Flask(__name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_depression_ensemble_v3_optimized.pkl")


def get_model():
    if not hasattr(get_model, "_model"):
        get_model._model = joblib.load(MODEL_PATH)
    return get_model._model

FEATURE_NAMES = [
    'Gender', 'Age', 'City', 'Profession', 'Academic Pressure',
    'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
    'Sleep Duration', 'Dietary Habits', 'Degree',
    'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
    'Financial Stress', 'Family History of Mental Illness'
]


def engineer_features(df):
    df['Total_Pressure'] = df['Academic Pressure'] + df['Work Pressure']

    df['Pressure_Satisfaction_Ratio'] = (
        df['Total_Pressure'] /
        (df['Study Satisfaction'] + df['Job Satisfaction'] + 1)
    )

    df['Sleep_Deprived'] = df['Sleep Duration'].apply(
        lambda x: 1 if x == 'Less than 5 hours' else 0
    )

    df['Sleep_Quality'] = df['Sleep Duration'].map({
        'Less than 5 hours': 0,
        '5-6 hours': 1,
        '7-8 hours': 3,
        'More than 8 hours': 2
    })

    df['High_Risk'] = (
        (df['Have you ever had suicidal thoughts ?'] == 1) &
        (df['Financial Stress'] >= 4)
    ).astype(int)

    df['Stress_Multiplier'] = (
        df['Academic Pressure'] * df['Work Pressure']
    )

    df['Overall_Satisfaction'] = (
        df['Study Satisfaction'] + df['Job Satisfaction']
    )

    df['Satisfaction_Pressure_Ratio'] = (
        (df['Study Satisfaction'] + df['Job Satisfaction'] + 1) /
        (df['Total_Pressure'] + 1)
    )

    df['WorkLife_Balance'] = (
        df['Overall_Satisfaction'] / (df['Work/Study Hours'] + 1)
    )

    df['FinAcad_Stress_Interaction'] = (
        df['Financial Stress'] * df['Academic Pressure']
    )

    df['Suicide_Risk_Severity'] = (
        df['Have you ever had suicidal thoughts ?'] *
        (df['Financial Stress'] + df['Total_Pressure']) / 2
    )

    df['MH_Vulnerability'] = (
        df['Have you ever had suicidal thoughts ?'] * 0.3 +
        (df['Financial Stress'] / 5) * 0.2 +
        (df['Total_Pressure'] / 5) * 0.2 +
        df['Sleep_Deprived'] * 0.15 +
        (1 - df['Overall_Satisfaction'] / 10) * 0.15
    )

    df['CGPA_Performance_Gap'] = (
        (4.0 - df['CGPA']) * df['Academic Pressure']
    )

    df['Perfect_Health'] = (
        (df['Have you ever had suicidal thoughts ?'] == 0) &
        (df['Financial Stress'] <= 2) &
        (df['Sleep_Deprived'] == 0) &
        (df['Total_Pressure'] <= 2)
    ).astype(int)

    df['Extreme_Stress'] = (
        ((df['Academic Pressure'] >= 4).astype(int) +
         (df['Financial Stress'] >= 4).astype(int) +
         (df['Work/Study Hours'] >= 10).astype(int) +
         df['Sleep_Deprived'] +
         (df['Overall_Satisfaction'] <= 3).astype(int))
        >= 3
    ).astype(int)

    df['CGPA_Achievement'] = (
        df['CGPA'] / (df['Academic Pressure'] + 1)
    )

    return df


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        missing = [f for f in FEATURE_NAMES if f not in request.form]
        if missing:
            return render_template("result.html", error="Missing required field(s): " + ", ".join(missing))

        data = {}
        for field in FEATURE_NAMES:
            raw = request.form[field].strip()
            if not raw:
                return render_template("result.html", error=f"Field '{field}' is empty.")
            data[field] = raw

        NUMERIC_RANGES = {
            'Gender': (int, 0, 1),
            'Age': (int, 10, 50),
            'Academic Pressure': (int, 1, 5),
            'Work Pressure': (int, 1, 5),
            'CGPA': (float, 0.0, 4.0),
            'Study Satisfaction': (int, 1, 5),
            'Job Satisfaction': (int, 1, 5),
            'Work/Study Hours': (int, 0, 20),
            'Financial Stress': (int, 1, 5),
            'Have you ever had suicidal thoughts ?': (int, 0, 1),
            'Family History of Mental Illness': (int, 0, 1),
        }

        for field, (cast, lo, hi) in NUMERIC_RANGES.items():
            try:
                val = cast(data[field])
            except (ValueError, TypeError):
                return render_template("result.html", error=f"'{field}' must be a number.")
            if not (lo <= val <= hi):
                return render_template("result.html", error=f"'{field}' must be between {lo} and {hi}.")
            data[field] = val

        mdl = get_model()
        input_df = pd.DataFrame([data])
        input_df = engineer_features(input_df)

        prediction = mdl.predict(input_df)[0]
        probability = mdl.predict_proba(input_df)[0]

        no_dep_prob = round(probability[0] * 100, 2)
        dep_prob = round(probability[1] * 100, 2)

        return render_template(
            "result.html",
            prediction="Depression detected" if prediction == 1 else "No depression detected",
            dep_prob=dep_prob,
            no_dep_prob=no_dep_prob,
            risk_level=risk_level,
            advice=advice
        )

    except Exception as e:
        return render_template("result.html", error="An unexpected error occurred. Please check your inputs and try again.")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/documentation")
def documentation():
    return render_template("documentation.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        missing = [f for f in FEATURE_NAMES if f not in data]
        if missing:
            return jsonify({"error": "Missing required field(s): " + ", ".join(missing)}), 400

        for field in FEATURE_NAMES:
            raw = data.get(field)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                return jsonify({"error": f"Field '{field}' is empty."}), 400

        NUMERIC_RANGES = {
            'Gender': (int, 0, 1),
            'Age': (int, 10, 50),
            'Academic Pressure': (int, 1, 5),
            'Work Pressure': (int, 1, 5),
            'CGPA': (float, 0.0, 4.0),
            'Study Satisfaction': (int, 1, 5),
            'Job Satisfaction': (int, 1, 5),
            'Work/Study Hours': (int, 0, 20),
            'Financial Stress': (int, 1, 5),
            'Have you ever had suicidal thoughts ?': (int, 0, 1),
            'Family History of Mental Illness': (int, 0, 1),
        }

        for field, (cast, lo, hi) in NUMERIC_RANGES.items():
            try:
                val = cast(data[field])
            except (ValueError, TypeError):
                return jsonify({"error": f"'{field}' must be a number."}), 400
            if not (lo <= val <= hi):
                return jsonify({"error": f"'{field}' must be between {lo} and {hi}."}), 400
            data[field] = val

        mdl = get_model()
        input_df = pd.DataFrame([data])
        input_df = engineer_features(input_df)

        prediction = mdl.predict(input_df)[0]
        probability = mdl.predict_proba(input_df)[0]

        return jsonify({
            "prediction": int(prediction),
            "depression_probability": round(float(probability[1]), 4),
            "no_depression_probability": round(float(probability[0]), 4)
        })
    except Exception as e:
        return jsonify({"error": "An internal error occurred processing the request."}), 500


@app.route("/health")
def health():
    model_ok = False
    model_error = None
    try:
        mdl = get_model()
        model_ok = True
    except Exception as e:
        model_error = str(e)
    return jsonify({
        "status": "ok",
        "model_loaded": model_ok,
        "model_error": model_error
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
