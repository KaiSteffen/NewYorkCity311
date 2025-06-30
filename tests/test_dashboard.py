# tests/test_dashboard.py
import pytest
import pandas as pd
import sys
from pathlib import Path
import os
import json
import pandas as pd
import locale

# Füge das 'src'-Verzeichnis zum Python-Pfad hinzu
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))


# Fixture zum Laden des Fairness-Metrik-Datensatzes aus der Ergebnisdatei
@pytest.fixture
def fairness_df():
    df = pd.read_csv("results/fairness_metrics.csv")
    return df

# Testet, ob die Fairness-Metrik-Datei die erwarteten Spalten enthält
# (z.B. Klasse und DemographicParityDifference)
def test_fairness_file_loads(fairness_df):
    assert "Klasse" in fairness_df.columns
    assert "DemographicParityDifference" in fairness_df.columns

# Testet, ob die Werte für DemographicParityDifference im erwarteten Bereich liegen
# und ob die Spalte keine ausschließlich ungültigen Werte enthält
def test_metric_ranges(fairness_df):
    dpd_vals = pd.to_numeric(fairness_df["DemographicParityDifference"], errors="coerce")
    assert dpd_vals.isna().sum() < len(dpd_vals)  # Es gibt mindestens einen gültigen Wert
    assert dpd_vals.max() < 1.0  # Maximalwert liegt unter 1.0
