import pytest
import pandas as pd
from fraud_detector import FraudDetectionModel

@pytest.fixture
def loaded_pipeline() -> FraudDetectionModel :
    """Loads the trained model artifact. Acts like a JUnit @Before setup"""
    return FraudDetectionModel.load_model('fraud_detector_model.pkl')

def test_predict_proba_valid_input(loaded_pipeline : FraudDetectionModel) -> None :
    """Ensure the model outputs a valid probability (0.0 to 1.0) for standard data."""

    # Create a single row of dummy transaction data (30 columns matching Kaggle data)
    dummy_data : pd.DataFrame = pd.DataFrame(
        [[0] * 30],
        columns = ['Time'] + [f'V{i}' for i in range (1,29)] + ['Amount']
    )

    # Get the fraud probability
    prob : np.ndarray = loaded_pipeline.predict_proba(dummy_data)

    # Assert the output is a valid mathematical probability
    assert 0.0 <= prob[0] <= 1.0

    
def test_predict_proba_invalid_input(loaded_pipeline : FraudDetectionModel) -> None :
    """Ensure the model safely rejects invalid data types (like strings)."""

    # Pass a string ('one_hundred_dollars') instead of a float for the Amount
    bad_data : pd.DataFrame = pd.DataFrame(
        [[0] * 29 + ['one_hundred_dollars']], 
        columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    )

    # We assert that XGBoost correctly raises an error rather than silently failing
    with pytest.raises((ValueError, TypeError)):
        loaded_pipeline.predict_proba(bad_data)