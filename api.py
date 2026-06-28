from fastapi import FastAPI, HTTPException
from fraud_detector import FraudDetectionModel
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd

#Initialising the application
app : FastAPI = FastAPI(title="Real Time Fraud Detection API", version="1.0")

#Loading the model singleton on startup
print("Loading model artifact into memory...")
pipeline : FraudDetectionModel = FraudDetectionModel.load_model('fraud_detector_model.pkl')
print("Model loaded and ready for inference")

#DTO Object 
class TransactionInput(BaseModel) :
    Time: float = Field(... , description="Seconds elapsed since the first transaction")
    features: Dict[str,float] = Field(..., description="Dictionary containing keys 'V1' through 'V28'")
    Amount: float = Field(..., description="Transaction amount in dollars")

@app.post('/predict')
def predict_fraud(transaction: TransactionInput) -> dict :
    """Receiving a transaction payload, reconstructing the data and calculating the risk score"""
    try : 
        row_data : dict = {'Time' : transaction.Time}
        row_data.update(transaction.features)
        row_data['Amount'] = transaction.Amount

        expected_columns : list = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

        df : pd.DataFrame = pd.DataFrame([row_data] , columns=expected_columns)

        risk_probability : float = pipeline.predict_proba(df)[0]

        return {
            "status" : "success",
            "risk_score" : float(risk_probability),
            "is_flagged" : bool(risk_probability >= 0.20)
        }
    except Exception as e :
        raise HTTPException(status_code=400 , detail=str(e))