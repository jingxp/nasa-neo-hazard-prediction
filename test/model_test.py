import pytest
import pandas as pd
import sys

sys.path.append('../training/')

from training_flow import read_data, data_prep, data_split

@pytest.fixture
def mock_df():
    return pd.read_csv('../training/nearest-earth-objects(1910-2024).csv')

def test_read_data():
    # Test reading data correctly
    df = read_data('../training/nearest-earth-objects(1910-2024).csv')
    print(df.columns)
    assert not df.empty
    assert list(df.columns) == ['neo_id', 'name', 'absolute_magnitude', 'estimated_diameter_min',
       'estimated_diameter_max', 'orbiting_body', 'relative_velocity',
       'miss_distance', 'is_hazardous']

def test_data_prep(mock_df):
    processed_df = data_prep(mock_df)
    assert "neo_id" not in processed_df.columns
    assert "name" not in processed_df.columns
    assert "orbiting_body" not in processed_df.columns
    assert "average_diameter" in processed_df.columns
    assert "scaled_relative_velocity" in processed_df.columns
    assert "momentum" in processed_df.columns
    assert "velocity_distance_ratio" in processed_df.columns
    assert "diameter_magnitude_ratio" in processed_df.columns
    assert "hazardous_label" in processed_df.columns
    assert "is_hazardous" not in processed_df.columns

def test_data_split(mock_df):
    processed_df = data_prep(mock_df)
    X_train, X_test, y_train, y_test = data_split(processed_df)
    
    # Check dimensions
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    
    # Ensure the split is 75/25
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) > len(X_test)
    assert len(y_train) > len(y_test)

if __name__ == "__main__":
    pytest.main()