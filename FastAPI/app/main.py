from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import mlflow
import numpy as np
import logging
import os

# Import our custom modules
from src.model.explainability import AMLExplainer
from shap_translator import translate_shap_for_llm
from llm_service import generate_investigation_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AML Triage API", version="1.0")

#global variables to hold loaded model artifacts
explainer = None
feature_names = None

class TransactionData(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    probability: float
    is_alert: bool
    threshold_used: float

class InvestigationResponse(BaseModel):
    shap_raw: List[Dict[str, float]]
    evidence_list: str
    llm_summary: str

@app.on_event("startup")
def load_model_and_explainer():
    """
    fetches the production model from MLflow on startup.
    """
    global explainer, feature_names
    
    
    # Set MLflow tracking URI to Dagshub
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/virajdeshmukh080818/AML.mlflow"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    # Load model from registry (Replace with your actual registered model path)
    model_uri = "models:/AML_LightGBM_Production/1" 
    logger.info(f"Loading model from MLflow: {model_uri}")
    
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Extract feature names from model signature
    feature_names = model.metadata.signature.input_columns.schema.columns()
    # Note: Depending on how mlflow.pyfunc serializes it, you might need to hardcode 
    # the features list here if signature parsing fails.
    
    explainer = AMLExplainer(model=model._model_impl.python_model.model, feature_names=feature_names)
    logger.info("Model and SHAP Explainer loaded successfully.")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionData):
    """
    Fast prediction endpoint. Does NOT call the LLM.
    """
    # Format input
    X_input = np.array([[transaction.features.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    
    #predict (assuming binary classification)
    prob = explainer.model.predict_proba(X_input)[0][1]
    
    #using your Best F1 threshold as the alert trigger
    threshold = 0.20 
    
    return {
        "probability": float(prob),
        "is_alert": prob > threshold,
        "threshold_used": threshold
    }

@app.post("/investigate", response_model=InvestigationResponse)
def investigate_alert(transaction: TransactionData):
    """
    deep dive endpoint. Computes SHAP, translates it, and calls LLM.
    called by the frontend ONLY when an analyst clicks Investigate on an alert.
    """
    if not explainer:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    #get SHAP values
    shap_data = explainer.explain_transaction(transaction.features)
    
    #translate to English
    human_readable_evidence = translate_shap_for_llm(shap_data)
    
    #send to LLM
    narrative = generate_investigation_summary(human_readable_evidence)
    
    return {
        "shap_raw": shap_data,             
        "evidence_list": human_readable_evidence, 
        "llm_summary": narrative  
    }