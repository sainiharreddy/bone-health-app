import os
import logging
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)

def fetch_raw_clinical_data():
    """Simulates an orthopedic research registry tracking bone health markers."""
    np.random.seed(101)
    n_samples = 2000
    
    data = {
        'age': np.random.randint(40, 95, size=n_samples),
        'bmi': np.random.normal(loc=26.5, scale=5.0, size=n_samples),
        't_score_bmd': np.random.normal(loc=-1.5, scale=1.2, size=n_samples),
        'prior_fracture': np.random.choice([1, 0], size=n_samples, p=[0.15, 0.85]),
        'rheumatoid_arthritis': np.random.choice([1, 0], size=n_samples, p=[0.08, 0.92]),
        'current_smoker': np.random.choice([1, 0], size=n_samples, p=[0.22, 0.78]),
        'is_female': np.random.choice([1, 0], size=n_samples, p=[0.6, 0.4]),
        'fracture_occurred': np.random.choice([1, 0], size=n_samples, p=[0.12, 0.88])
    }
    return pd.DataFrame(data)

def serialize_balanced_pipeline():
    """Trains a high-performance tree model with balanced weights and saves it."""
    df = fetch_raw_clinical_data()
    
    X = df.drop(columns=['fracture_occurred'])
    y = df['fracture_occurred']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)
    
    logging.info("Training high-efficiency Gradient Boosting model (XGBoost)...")
    
    # Calculate balance multiplier (negative cases divided by positive cases)
    scale_weight = float(sum(y_train == 0) / sum(y_train == 1))
    
    # Initialize perfectly balanced model
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_weight, random_state=101)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    logging.info("\n" + classification_report(y_test, predictions))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump((model, X_train), "models/advanced_fracture_model.pkl")
    logging.info("Production pipeline saved to models/advanced_fracture_model.pkl")

if __name__ == "__main__":
    serialize_balanced_pipeline()
