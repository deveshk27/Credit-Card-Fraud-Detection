import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc

class FraudDetectionModel :
    def __init__(self, random_state: int = 42) -> None:
        self.random_state : int = random_state
        self.model : XGBClassifier = XGBClassifier(random_state=random_state, eval_metric='logloss')
        self.smote : SMOTE = SMOTE(random_state=random_state)

    def fit_resample(self, X_train : pd.DataFrame , y_train : pd.Series) -> None :
        """Applies SMOTE and trains the XGBoost Model"""
        print("Applying SMOTE geometric interpolation")
        X_train_res , y_train_res = self.smote.fit_resample(X_train, y_train)

        print("Training sequential trees (XGBoost)")
        self.model.fit(X_train_res, y_train_res)
        print("Pipeline training completed successfully")

    def predict_proba(self, X_test : pd.DataFrame) -> np.ndarray:
        """Returns raw probabilities of fraud transactions (Class 1)"""
        return self.model.predict_proba(X_test)[:, 1]

    def evaluate(self, X_test : pd.DataFrame , y_test : pd.Series ) -> float :
        """Evaluates model using PR-AUC and a classification report."""
        y_pred_proba : np.ndarray = self.predict_proba(X_test)

        custom_threshold : float = 0.20
        y_pred : np.ndarray = (y_pred_proba >= custom_threshold).astype(int)

        print(f"\n--- Classification Report ({int(custom_threshold*100)}% Threshold) ---")
        print(classification_report(y_test, y_pred))

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc_score : float = auc(recall, precision)

        print(f"PR-AUC Score: {pr_auc_score:.4f}")
        return pr_auc_score

    @staticmethod
    def load_model(file_path='fraud_detector_model.pkl') -> 'FraudDetectionModel':
        return joblib.load(file_path)

