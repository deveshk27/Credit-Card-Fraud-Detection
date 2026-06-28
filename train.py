import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Importantly, we IMPORT the class so its true namespace ('fraud_detector') is preserved
from fraud_detector import FraudDetectionModel

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

print("Initializing and training pipeline...")
pipeline : FraudDetectionModel = FraudDetectionModel()
pipeline.fit_resample(X_train, y_train)

print("Saving artifact to disk...")
joblib.dump(pipeline, 'fraud_detector_model.pkl')
print("Done! Model saved cleanly with the correct namespace.")