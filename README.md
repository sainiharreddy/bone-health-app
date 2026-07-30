# Explainable Clinical Diagnostics (XAI) Engine: Bone Fracture Risk Pipeline

A production-grade, low-latency machine learning microservice built to ingest multimodal patient biomarkers, predict continuous statistical probabilities of future bone fractures, and instantly expose localized game-theory feature attributions via an asynchronous API gateway. 

This architecture addresses the critical regulatory and compliance hurdles in modern healthtech by eliminating "black-box" algorithmic decision-making, providing point-of-care clinicians with fully transparent, auditable medical justifications.

---

## 🚀 Core Features
- **Stateless Inference Gateway:** Implemented asynchronous REST endpoints using **FastAPI** to deliver rapid, low-latency prediction modeling.
- **Runtime Border Validation:** Enforced total pipeline data safety using strict **Pydantic** validation schemas, intercepting and filtering out malformed payloads prior to the model layer.
- **Imbalanced Class Mitigation:** Solved clinical dataset skew (minority fracture class anomalies) by engineering dynamic POS class-weight scaling metrics directly into an **XGBoost Classifier**.
- **Explainable AI Compliance (XAI):** Integrated high-speed **SHAP (Shapley Additive exPlanations)** matrices inside the post-processing engine to calculate and expose directional numerical impact scores for every biomarker feature.

---

## 📁 Repository Structure
```text
bone-health-app/
│
├── app/
│   └── main.py             # FastAPI microservice routing & SHAP extraction logic
│
├── models/
│   └── advanced_fracture_model.pkl # Serialized class-balanced XGBoost pipeline
│
├── train.py                # Deterministic registry pipeline & model serialization
├── requirements.txt        # Enterprise dependency blueprint manifest
└── README.md               # Production architecture documentation
```

---

## 🩺 System Inputs & Predictive Variables
The internal predictive layers evaluate seven core clinical, structural, and behavioral markers:
- `age`: Patient chronological age (validated scale range: 40 - 110)
- `bmi`: Calculated Body Mass Index (validated scale range: 10.0 - 60.0)
- `t_score_bmd`: Bone Mineral Density (BMD) deviation marker relative to young adult baseline scales
- `prior_fracture`: Historical structural trauma indicator (Binary: `0` for None, `1` for Yes)
- `rheumatoid_arthritis`: Chronic inflammatory comorbidity status (Binary: `0` for Negative, `1` for Positive)
- `current_smoker`: Behavioral toxicological indicator (Binary: `0` for Non-Smoker, `1` for Active Smoker)
- `is_female`: Biological sex marker mapping structural demographic risk splits (Binary: `0` for Male, `1` for Female)

---

## 📊 Live Sample API Payloads & Outputs

### Endpoints
* **`POST /predict/fracture_risk`**

### Sample Input Payload
```json
{
  "age": 81,
  "bmi": 18.5,
  "t_score_bmd": -2.9,
  "prior_fracture": 1,
  "rheumatoid_arthritis": 1,
  "current_smoker": 1,
  "is_female": 1
}
```

### Production JSON API Response Body
```json
{
  "calculated_fracture_probability": 0.4037,
  "clinical_risk_tier": "Moderate Risk",
  "primary_risk_driver_variable": "bmi",
  "explainable_ai_metric_contributions": {
    "age": -0.1859,
    "bmi": 0.1375,
    "t_score_bmd": 0.007,
    "prior_fracture": -0.3717,
    "rheumatoid_arthritis": -0.038,
    "current_smoker": 0.0628,
    "is_female": -0.0035
  },
  "clinical_documentation_note": "Positive SHAP scores (+) increase fracture risk; negative scores (-) are protective."
}
```

---

## 🛠️ Technology Stack & Engineering Libraries
- **Language:** Python 3.11+
- **Serving Architecture:** FastAPI, Uvicorn, ASGI Loop Frameworks
- **Machine Learning Core:** XGBoost, Scikit-Learn
- **Model Transparency:** SHAP (TreeExplainer Game-Theory Framework)
- **Data Schemas & Types:** Pydantic Core v2, Pandas, NumPy
- **Persistence Management:** Joblib Binary Deserializers
