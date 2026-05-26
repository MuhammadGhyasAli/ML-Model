# Depression Risk Screening — ML-Powered Web Application

## Project Overview

A production-grade clinical depression risk screening tool built with an ensemble machine learning pipeline served through a Flask web interface. The application accepts 16 raw input features across demographic, academic, and lifestyle categories, engineers 16 additional derived features, and returns a depression risk probability via a soft-voting ensemble of four classifiers.

---

## 1. Project Structure

```
E:\Student Depression\
├── app.py                                          # Flask application (261 lines)
├── student_depression_ensemble_v3_optimized.pkl    # Trained ML pipeline (62.5 MB)
├── templates/
│   ├── index.html          # Assessment form (1504 lines, 49 KB)
│   ├── result.html         # Prediction result (459 lines, 19 KB)
│   ├── about.html          # Model architecture (292 lines, 15 KB)
│   └── documentation.html  # API reference (360 lines, 20 KB)
├── Student Depression Dataset - Cleaned.csv         # Training data (2196 KB)
├── Student Depression.ipynb                         # Model training notebook
├── *.png                                            # Training visualizations
└── PROJECT_REPORT.md                                # This document
```

---

## 2. Machine Learning Pipeline

### 2.1 Training Data

- **Source**: Student Depression Dataset (Kaggle)
- **Samples**: ~27,900 clinical records
- **Target**: Binary depression classification (depressed / not depressed)

### 2.2 Pipeline Architecture (`sklearn.pipeline.Pipeline`)

```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num (11 raw numeric + 16 engineered → 27 columns)
│   │   ├── SimpleImputer(strategy='median')
│   │   └── StandardScaler()
│   ├── ord (1 column: Sleep Duration)
│   │   ├── SimpleImputer(strategy='most_frequent')
│   │   └── OrdinalEncoder(handle_unknown='use_encoded_value')
│   └── nom (4 columns: City, Profession, Dietary Habits, Degree)
│       ├── SimpleImputer(strategy='most_frequent')
│       └── OneHotEncoder(handle_unknown='ignore')
└── classifier (VotingClassifier, soft voting)
    ├── LogisticRegression      (C=0.1, solver=lbfgs)          weight=1.0
    ├── RandomForestClassifier  (n=400, max_depth=12)           weight=2.0
    ├── XGBClassifier           (n=200, max_depth=4, lr=0.05)   weight=2.0
    └── GradientBoostingClassifier (n=300, max_depth=5, lr=0.05) weight=1.5
```

**Total features passed to classifier**: 27 (numeric) + ordinal + one-hot encoded (varies by unique categories)

### 2.3 Raw Input Features (16)

| # | Name | Type | Range |
|---|------|------|-------|
| 1 | Gender | Binary | 0=Female, 1=Male |
| 2 | Age | Integer | 10–50 |
| 3 | City | Text | Free entry |
| 4 | Profession | Hidden | Always "Student" |
| 5 | Academic Pressure | Integer | 1–5 (Very Low–Very High) |
| 6 | Work Pressure | Integer | 1–5 (Very Low–Very High) |
| 7 | CGPA | Float | 0.00–4.00 |
| 8 | Study Satisfaction | Integer | 1–5 (Very Dissatisfied–Very Satisfied) |
| 9 | Job Satisfaction | Integer | 1–5 (Very Dissatisfied–Very Satisfied) |
| 10 | Sleep Duration | Categorical | Less than 5h / 5-6h / 7-8h / More than 8h |
| 11 | Dietary Habits | Categorical | Healthy / Moderate / Unhealthy / Others |
| 12 | Degree | Text | Free entry |
| 13 | Have you ever had suicidal thoughts ? | Binary | 0=No, 1=Yes |
| 14 | Work/Study Hours | Integer | 0–20 |
| 15 | Financial Stress | Integer | 1–5 (None–Severe) |
| 16 | Family History of Mental Illness | Binary | 0=No, 1=Yes |

### 2.4 Engineered Features (16)

| Feature | Formula | Purpose |
|---------|---------|---------|
| Total_Pressure | Academic Pressure + Work Pressure | Cumulative stress load |
| Pressure_Satisfaction_Ratio | Total_Pressure / (Study Sat + Job Sat + 1) | Pressure relative to satisfaction |
| Sleep_Deprived | 1 if Sleep Duration = "Less than 5 hours" | Binary deprivation flag |
| Sleep_Quality | Ordinal map of sleep duration | Quality score (0-3) |
| High_Risk | Suicidal thoughts AND Financial Stress >= 4 | High-risk interaction |
| Stress_Multiplier | Academic Pressure × Work Pressure | Compounding stress |
| Overall_Satisfaction | Study Sat + Job Sat | Total satisfaction |
| Satisfaction_Pressure_Ratio | (Study Sat + Job Sat + 1) / (Total_Pressure + 1) | Inverse satisfaction-pressure |
| WorkLife_Balance | Overall_Satisfaction / (Hours + 1) | Balance metric |
| FinAcad_Stress_Interaction | Financial Stress × Academic Pressure | Double stress interaction |
| Suicide_Risk_Severity | Suicidal thoughts × (Financial + Total_Pressure) / 2 | Weighted risk severity |
| MH_Vulnerability | Weighted composite of 5 risk factors | Multi-factor vulnerability |
| CGPA_Performance_Gap | (4.0 - CGPA) × Academic Pressure | Academic distress |
| Perfect_Health | All favorable conditions | Absence of risk markers |
| Extreme_Stress | ≥3 of 5 extreme conditions | Multi-domain extreme stress |
| CGPA_Achievement | CGPA / (Academic Pressure + 1) | Performance under pressure |

### 2.5 Performance Metrics (from training notebook)

| Metric | Value |
|--------|-------|
| AUC-ROC | 99.2% |
| Training samples | ~27,900 |
| Cross-validation | Stratified K-fold |
| Voting strategy | Soft (probability-weighted) |
| Class weights | [1.0, 2.0, 2.0, 1.5] |

---

## 3. Web Application Architecture

### 3.1 Stack

- **Backend**: Python 3.13, Flask 3.x
- **Frontend**: HTML5, CSS3 (vanilla, 1500+ lines), JavaScript (vanilla)
- **Model**: scikit-learn 1.x, XGBoost
- **Serialization**: joblib

### 3.2 Routes

| Route | Method | Description | Response |
|-------|--------|-------------|----------|
| `/` | GET | Assessment form | `index.html` |
| `/predict` | POST | Form submit → prediction | `result.html` |
| `/about` | GET | Model architecture & metrics | `about.html` |
| `/documentation` | GET | API reference & usage | `documentation.html` |
| `/api/predict` | POST | JSON API → prediction | JSON |

### 3.3 Request Flow

```
User → / → index.html (form)
     → POST /predict → validate fields → engineer_features()
     → model.predict() → return result.html
     → /api/predict (JSON) → same pipeline → JSON response
```

### 3.4 Validation

- **Client-side**: Real-time field validation (age, CGPA, hours range checks), progress tracking
- **Server-side**: All 16 fields checked for presence, type, and numeric range limits

### 3.5 Risk Classification

| Depression Probability | Risk Level |
|----------------------|------------|
| < 25% | VERY LOW RISK |
| 25% – 49% | LOW RISK |
| 50% – 74% | MODERATE RISK |
| ≥ 75% | HIGH RISK |

---

## 4. User Interface

### 4.1 Design System

- **Theme**: Light enterprise dashboard
- **Colors**: Indigo primary (#6366f1), slate text (#1e293b), success/danger/warning semantic colors
- **Typography**: Inter (Google Fonts), 300–700 weights
- **Layout**: Fixed sidebar (260px) + fluid main content area
- **Responsiveness**: 4 breakpoints (desktop / 1024px / 768px / 480px)
- **Sidebar navigation**: Assessment, About the Model, Documentation — active state on each page

### 4.2 Form Pages

**`index.html`** — 3-section assessment form:
1. **Personal Information**: Gender (select), Age (number), City (text), Profession (hidden = "Student")
2. **Academic & Work**: Degree (text), CGPA (number), pressure/satisfaction radio groups (1–5), hours (number)
3. **Lifestyle & Health**: Sleep Duration (select), Dietary Habits (select), Financial Stress (radio), Suicidal Ideation (select), Family History (select)

Key UI features:
- Progress bar (tracks 15 visible fields, handles radio groups as 1)
- Model banner showing ensemble architecture and performance
- Google Forms-style radio button groups with circle indicators
- Real-time validation with success/error indicators
- Loading overlay with animated progress during submission
- localStorage auto-save (disabled by default, ready to enable)
- Mobile sidebar toggle (fixed hamburger button)
- Print-friendly result page

### 4.3 Result Page

**`result.html`** — Displays:
- Primary outcome (Depression detected / No depression detected)
- Risk level badge (color-coded: red/orange/green)
- Animated probability bar
- Model confidence & risk classification metadata
- Contextual advice list (clinical recommendations or preventive guidance)
- Actions: New Assessment / Print Report

### 4.4 Information Pages

- **`about.html`**: Ensemble model architecture, 4-model details, performance metrics, feature engineering explanation, limitations
- **`documentation.html`**: Web interface guide, REST API reference (complete JSON schema, curl example), input field table, risk classification table, validation rules

### 4.5 Responsive Breakpoints

| Screen Width | Behavior |
|-------------|----------|
| > 1024px | 2-column form grid, sidebar pinned |
| 768–1024px | Reduced padding, model-divider hidden |
| < 768px | Sidebar slides off-screen, hamburger toggle, 1-column form, radio options shrink |
| < 480px | Compact model banner, tighter radio groups, smaller text |

---

## 5. Deployment

### 5.1 Local

```bash
pip install flask pandas numpy scikit-learn xgboost joblib
python app.py
# → http://localhost:5000
```

### 5.2 Environment Variables

- `PORT` — Server port (default: 5000)

### 5.3 Dependencies

- flask, pandas, numpy, scikit-learn, xgboost, joblib

---

## 6. API Reference

### `POST /api/predict`

**Request** (JSON):
```json
{
  "Gender": 1,
  "Age": 22,
  "City": "Karachi",
  "Profession": "Student",
  "Academic Pressure": 3,
  "Work Pressure": 2,
  "CGPA": 3.2,
  "Study Satisfaction": 4,
  "Job Satisfaction": 3,
  "Sleep Duration": "7-8 hours",
  "Dietary Habits": "Moderate",
  "Degree": "BSc Computer Science",
  "Have you ever had suicidal thoughts ?": 0,
  "Work/Study Hours": 8,
  "Financial Stress": 2,
  "Family History of Mental Illness": 0
}
```

**Response** (200):
```json
{
  "prediction": 0,
  "depression_probability": 0.1308,
  "no_depression_probability": 0.8692
}
```

**Error** (400/500): `{ "error": "descriptive message" }`

---

## 7. Development Notes

- The `OneHotEncoder` uses `handle_unknown='ignore'` — unseen City/Degree/Profession values produce a zero vector rather than errors.
- The `OrdinalEncoder` for Sleep Duration uses `handle_unknown='use_encoded_value'` for robustness.
- Feature names with spaces (e.g., `"Have you ever had suicidal thoughts ?"`) are preserved exactly to match the training data column names.
- The `submit-bar` at the form bottom displays the engineered feature count ("32 clinical features including 16 engineered indicators") to communicate model sophistication to users.
- All templates include a `<meta name="viewport">` tag and a consistent CSS custom property system for maintainability.
