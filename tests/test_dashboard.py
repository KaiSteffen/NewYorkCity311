# tests/test_dashboard.py
import pytest
import pandas as pd

@pytest.fixture
def fairness_df():
    df = pd.read_csv("results/fairness_metrics.csv")
    return df

def test_fairness_file_loads(fairness_df):
    assert "Klasse" in fairness_df.columns
    assert "DemographicParityDifference" in fairness_df.columns

def test_metric_ranges(fairness_df):
    dpd_vals = pd.to_numeric(fairness_df["DemographicParityDifference"], errors="coerce")
    assert dpd_vals.isna().sum() < len(dpd_vals)
    assert dpd_vals.max() < 1.0
