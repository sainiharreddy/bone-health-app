import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import shap

print("Initializing FastAPI App...")
app = FastAPI(title="Explainable Clinical Diagnostics (XAI Engine)", version="2.0.0")

print("Loading trained XGBoost model asset...")
MODEL_PATH = "models/advanced_fracture_model.pkl" if os.path.exists("models/advanced_fracture_model.pkl") else "/app/models/advanced_fracture_model.pkl"
model, X_train = joblib.load(MODEL_PATH)

print("Initializing high-speed SHAP explainer matrix...")
# Optimize sample size for instant startup
bg_sample = X_train.sample(10, random_state=42)
explainer = shap.TreeExplainer(model, bg_sample)

class PatientBiomarkers(BaseModel):
    age: int = Field(..., ge=40, le=110)
    bmi: float = Field(..., ge=10.0, le=60.0)
    t_score_bmd: float = Field(..., ge=-5.0, le=3.0)
    prior_fracture: int = Field(..., ge=0, le=1)
    rheumatoid_arthritis: int = Field(..., ge=0, le=1)
    current_smoker: int = Field(..., ge=0, le=1)
    is_female: int = Field(..., ge=0, le=1)

@app.post("/predict/fracture_risk")
def calculate_diagnostic_risk(patient: PatientBiomarkers):
    features = ['age', 'bmi', 't_score_bmd', 'prior_fracture', 'rheumatoid_arthritis', 'current_smoker', 'is_female']
    patient_df = pd.DataFrame([[
        patient.age, patient.bmi, patient.t_score_bmd, 
        patient.prior_fracture, patient.rheumatoid_arthritis, 
        patient.current_smoker, patient.is_female
    ]], columns=features)
    
    probability = float(model.predict_proba(patient_df)[:, 1])
    shap_values = explainer.shap_values(patient_df)
    
    if isinstance(shap_values, list):
        raw_impact_scores = shap_values
    elif len(shap_values.shape) == 3:
        raw_impact_scores = shap_values[0, :, 1]
    else:
        raw_impact_scores = shap_values[0] if len(shap_values.shape) == 2 else shap_values
    
    xai_mapping = {
        name: round(float(score), 4) for name, score in zip(features, raw_impact_scores)
    }
    
    primary_driver = max(xai_mapping, key=xai_mapping.get)
    
    if probability > 0.65:
        tier = "Critical Diagnostic Fracture Risk"
        action = "Recommend prompt bone-density pharmaceutical intervention and immediate fall prevention mapping."
    elif probability > 0.30:
        tier = "Moderate Risk Profile"
        action = "Initiate proactive supplementary therapeutic care (Calcium/Vit D) and schedule a follow-up DEXA diagnostic scan."
    else:
        tier = "Low Routine Baseline Risk"
        action = "Maintain general preventative health monitoring protocols."

    return {
        "calculated_fracture_probability": round(probability, 4),
        "clinical_risk_tier": tier,
        "primary_risk_driver_variable": primary_driver,
        "explainable_ai_metric_contributions": xai_mapping,
        "clinical_documentation_note": "Positive SHAP numbers (+) signify factors extending fracture risk, while negative numbers (-) protect the baseline score."
    }

if __name__ == "__main__":
    import uvicorn
    print("Launching Uvicorn server production loop...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

