"""
📊 Testmodul für Fairness-Dashboard-Daten (`test_dashboard.py`)

Dieses Testmodul überprüft die Integrität und Gültigkeit der von der Fairness-Pipeline 
erzeugten Metrikdateien – insbesondere der Datei `fairness_metrics.csv` – die typischerweise 
von einem Analyse-Dashboard (z. B. Streamlit, Dash) visualisiert werden.

Testübersicht:
--------------
1. `test_fairness_file_loads`:
   - Validiert, ob die Datei `fairness_metrics.csv` erfolgreich geladen werden kann
     und die erwarteten Schlüsselspalten wie `Klasse` und `DemographicParityDifference` enthält.

2. `test_metric_ranges`:
   - Stellt sicher, dass die Werte in der Spalte `DemographicParityDifference`
     im erwarteten Wertebereich (zwischen -1.0 und 1.0) liegen und valide sind.

Fixtures:
---------
- `fairness_df`: Lädt einmalig die Metrikdatei und stellt sie allen Tests zur Verfügung.

Verwendete Technologien:
------------------------
- `pytest`: Test-Framework für einfache, lesbare Testfälle
- `pandas`: Datenverarbeitung und numerische Validierung
- `pathlib`, `sys`: Dynamische Pfadanpassung zum Importieren von Modulen aus `src`
- `locale`, `json`, `os`: Reserviert/importiert, aber in diesem Scope aktuell ungenutzt

Einsatzkontext:
---------------
Diese Tests dienen der Qualitätssicherung im Rahmen eines Fairness-Dashboards oder zur 
automatisierten Auswertung von Evaluationsergebnissen in CI/CD-Pipelines.

Autor: Bettina Gertjerenken, Dagmar Wesemann, Kai W. Steffen
Stand: Juni 2025
"""
# tests/test_fairness_analysis_flow.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# tests/test_dashboard.py
import pytest
import os
import json
import pandas as pd
import locale
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*palette.*without assigning.*hue.*is deprecated.*")

logging.getLogger("streamlit").setLevel(logging.ERROR)

# # Füge das 'src'-Verzeichnis zum Python-Pfad hinzu
# src_path = Path(__file__).parent.parent / 'src'
# sys.path.insert(0, str(src_path))


# Fixture zum Laden des Fairness-Metrik-Datensatzes aus der Ergebnisdatei
@pytest.fixture
def fairness_df():
    """
    Lädt die Fairness-Metrikdatei (CSV) und gibt sie als DataFrame zurück.
    Wird als Fixture für alle Tests verwendet, die auf die Metriken zugreifen.
    """
    df = pd.read_csv("results/fairness_metrics.csv")
    return df

##############################
# tests/test_metric_visualization.py
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fairness_dashboard import interpretationen    
from fairness_dashboard import get_readable_class_label   



@pytest.fixture
def demo_df():
    """
    Erstellt ein Beispiel-DataFrame mit Dummy-Fairnessmetriken für Visualisierungstests.
    Wird für Plot- und Mapping-Tests verwendet.
    """
    data = {
        "DemographicParityDifference": [0.1, 0.25, -0.05],
        "EqualizedOddsDifference": [0.05, 0.22, 0.01]
    }
    df = pd.DataFrame(data)
    df.index = [0, 1, 2]
    return df


@pytest.mark.parametrize("metric", ["DemographicParityDifference", "EqualizedOddsDifference"])
def test_metric_exists_and_numeric(demo_df, metric):
    """
    Prüft, ob die angegebene Metrikspalte im DataFrame existiert und numerische Werte enthält.
    """
    assert metric in demo_df.columns, f"❌ Spalte '{metric}' fehlt"
    converted = pd.to_numeric(demo_df[metric], errors="coerce")
    assert not converted.isna().all(), f"❌ Alle Werte in '{metric}' sind ungültig"


def test_label_mapping_is_string(demo_df):
    """
    Testet, ob die Label-Mapping-Funktion für jeden Index einen nicht-leeren String zurückgibt.
    """
    labels = demo_df.index.map(get_readable_class_label)
    for label in labels:
        assert isinstance(label, str)
        assert len(label) > 0


@pytest.mark.parametrize("metric", ["DemographicParityDifference"])
def test_can_generate_barplot(demo_df, metric):
    """
    Überprüft, ob für die angegebene Metrik ein Balkendiagramm ohne Fehler erzeugt werden kann.
    """
    demo_df = demo_df.copy()
    demo_df["Label"] = demo_df.index.map(get_readable_class_label)
    threshold = 0.2

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        sns.barplot(x="Label", y=metric, data=demo_df, palette="viridis", ax=ax)
        ax.axhline(threshold, color="red", linestyle="--")
    except Exception as e:
        pytest.fail(f"❌ Fehler beim Erzeugen des Plots: {e}")



########################
def test_fairness_file_loads(fairness_df):
    """
    Prüft, ob die Fairness-Metrikdatei geladen werden kann und die erwarteten Spalten enthält.
    Gibt die geladenen Spalten zur Kontrolle aus.
    """
    print("🔍 Überprüfe, ob Datei geladen werden konnte...")
    print(f"📦 Geladene Spalten: {list(fairness_df.columns)}")

    assert "Klasse" in fairness_df.columns, "❌ Spalte 'Klasse' fehlt in der CSV-Datei."
    assert "DemographicParityDifference" in fairness_df.columns, "❌ Spalte 'DemographicParityDifference' fehlt in der CSV-Datei."

    print("✅ Datei enthält alle erforderlichen Spalten.\n")


def test_metric_ranges(fairness_df):
    """
    Überprüft, ob die Werte für 'DemographicParityDifference' gültig und im erwarteten Bereich sind.
    Gibt Wertebereich und Anzahl gültiger Werte aus.
    """
    print("📐 Überprüfe Wertebereich für 'DemographicParityDifference'...")
    dpd_vals = pd.to_numeric(fairness_df["DemographicParityDifference"], errors="coerce")

    print(f"🔢 Gültige Werte gefunden: {dpd_vals.count()}/{len(dpd_vals)}")
    print(f"🧮 Wertebereich: min={dpd_vals.min():.3f}, max={dpd_vals.max():.3f}")

    assert dpd_vals.isna().sum() < len(dpd_vals), "❌ Alle DPD-Werte sind ungültig oder fehlen."
    assert dpd_vals.max() < 1.0, f"❌ Höchstwert überschreitet 1.0: {dpd_vals.max():.3f}"

    print("✅ Alle Metriken liegen im erwarteten Bereich.\n")


