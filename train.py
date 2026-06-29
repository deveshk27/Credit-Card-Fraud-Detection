import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from fraud_detector import FraudDetectionModel

# Importantly, we IMPORT the class so its true namespace ('fraud_detector') is preserved

print("Loading data...")
df: pd.DataFrame = pd.read_csv('creditcard.csv')
X: pd.DataFrame = df.drop(columns=['Class'])
y: pd.Series = df['Class']

X_train: pd.DataFrame
X_test: pd.DataFrame
y_train: pd.Series
y_test: pd.Series

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Fetching best params from MLflow...")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment = mlflow.get_experiment_by_name("Default")

best_run = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_auprc DESC"],
    max_results=1
).iloc[0]

best_params = {
    "n_estimators":     int(best_run["params.xgb__n_estimators"]),
    "max_depth":        int(best_run["params.xgb__max_depth"]),
    "learning_rate":    float(best_run["params.xgb__learning_rate"]),
    "subsample":        float(best_run["params.xgb__subsample"]),
    "colsample_bytree": float(best_run["params.xgb__colsample_bytree"]),
}
print(f"Best params: {best_params}")

# Training FraudDetectionModel with best params 
print("Initializing and training pipeline...")
pipeline: FraudDetectionModel = FraudDetectionModel(**best_params)
pipeline.fit_resample(X_train, y_train)

# Finding threshold using business-driven logic
print("Computing optimal threshold...")
y_pred_proba = pipeline.predict_proba(X_test)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

optimal_threshold = 0.45  # Can raise if too many false alerts, lower if missing too much fraud
idx = np.searchsorted(thresholds, optimal_threshold)
optimal_precision = precisions[idx]
optimal_recall    = recalls[idx]

print(f"Threshold: {optimal_threshold} → Precision: {optimal_precision:.4f} | Recall: {optimal_recall:.4f}")

# Attaching threshold and evaluating
pipeline.threshold = optimal_threshold
pipeline.evaluate(X_test, y_test)

# Saving artifact
joblib.dump(pipeline, 'fraud_detector_model.pkl')
print("Optimized model saved.")