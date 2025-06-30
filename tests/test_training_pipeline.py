import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
from pathlib import Path
import os

# Füge das 'src'-Verzeichnis zum Python-Pfad hinzu
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Importiere die zu testende Klasse
from training_pipeline import TrainingPipeline

@pytest.fixture
def mock_pycaret_and_save(monkeypatch):
    """Fixture, das alle externen und zeitaufwendigen Anrufe mockt."""
    # Mock für PyCaret-Funktionen
    mock_setup = MagicMock()

    # KORREKTUR: Erstelle einen detaillierteren Mock für das Modell,
    # damit die __class__.__name__ Kette funktioniert.
    # 1. Erstelle einen Mock für die *Klasse* des Modells
    mock_model_class = MagicMock()
    mock_model_class.__name__ = 'XGBClassifier'
    # 2. Erstelle einen Mock für die *Instanz* des Modells
    mock_model_instance = MagicMock()
    # 3. Weise dem Instanz-Mock die korrekte Klasse zu
    mock_model_instance.__class__ = mock_model_class
    
    # compare_models gibt jetzt unseren detaillierten Mock zurück
    mock_compare = MagicMock(return_value=mock_model_instance)
    
    mock_tune = MagicMock(return_value=MagicMock()) # Gibt ein anderes Mock-Modell zurück
    mock_finalize = MagicMock(return_value=MagicMock(get_params=lambda: {'model': MagicMock(save_model=MagicMock())})) # Simuliert die verschachtelte Struktur
    mock_save = MagicMock()
    
    monkeypatch.setattr('training_pipeline.setup', mock_setup)
    monkeypatch.setattr('training_pipeline.compare_models', mock_compare)
    monkeypatch.setattr('training_pipeline.tune_model', mock_tune)
    monkeypatch.setattr('training_pipeline.finalize_model', mock_finalize)
    monkeypatch.setattr('training_pipeline.save_model', mock_save)

    # Mock für Speicherfunktionen
    mock_joblib_dump = MagicMock()
    monkeypatch.setattr('training_pipeline.joblib.dump', mock_joblib_dump)
    
    # Gib die Mocks zurück, damit wir in den Tests darauf zugreifen können
    return {
        'setup': mock_setup,
        'compare_models': mock_compare,
        'tune_model': mock_tune,
        'finalize_model': mock_finalize,
        'save_model': mock_save,
        'joblib_dump': mock_joblib_dump
    }

@pytest.fixture
def sample_data(tmp_path):
    """Erstellt eine temporäre CSV-Datei mit Beispieldaten."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "sample_data.csv"
    
    # Erstelle ein DataFrame, das den echten Daten ähnelt
    df = pd.DataFrame({
        'Complaint_Type': ['HEAT/HOT WATER', 'Noise', 'HEAT/HOT WATER'],
        'duration_[days]': [1.0, 2.5, 0.5],
        'Feature1': [10, 20, 30],
        'Feature2': ['A', 'B', 'A']
    })
    df.to_csv(data_file, index=False)
    return str(data_file)

def test_pipeline_runs_without_tuning(mock_pycaret_and_save, sample_data, tmp_path):
    """Testet den kompletten Durchlauf der Pipeline OHNE Tuning."""
    model_dir = tmp_path / "models"
    
    pipeline = TrainingPipeline(data_path=sample_data, model_dir=str(model_dir))
    pipeline.run(skip_tuning=True)

    # Überprüfe, ob die wichtigsten Funktionen aufgerufen wurden
    mock_pycaret_and_save['setup'].assert_called_once()
    mock_pycaret_and_save['compare_models'].assert_called_once()
    mock_pycaret_and_save['finalize_model'].assert_called_once()
    mock_pycaret_and_save['save_model'].assert_called_once()
    mock_pycaret_and_save['joblib_dump'].assert_called_once()
    
    # Stelle sicher, dass Tuning NICHT aufgerufen wurde
    mock_pycaret_and_save['tune_model'].assert_not_called()

def test_pipeline_runs_with_tuning(mock_pycaret_and_save, sample_data, tmp_path):
    """Testet den kompletten Durchlauf der Pipeline MIT Tuning."""
    model_dir = tmp_path / "models"
    
    pipeline = TrainingPipeline(data_path=sample_data, model_dir=str(model_dir))
    pipeline.run(skip_tuning=False)

    # Überprüfe, ob die wichtigsten Funktionen aufgerufen wurden
    mock_pycaret_and_save['setup'].assert_called_once()
    mock_pycaret_and_save['compare_models'].assert_called_once()
    mock_pycaret_and_save['finalize_model'].assert_called_once()
    mock_pycaret_and_save['save_model'].assert_called_once()
    mock_pycaret_and_save['joblib_dump'].assert_called_once()
    
    # Stelle sicher, dass Tuning aufgerufen wurde
    mock_pycaret_and_save['tune_model'].assert_called_once()
    
def test_data_loading_error(tmp_path):
    """Testet, ob ein FileNotFoundError ausgelöst wird, wenn die Datendatei nicht existiert."""
    # Erwarte eine Exception vom Typ FileNotFoundError
    with pytest.raises(FileNotFoundError):
        pipeline = TrainingPipeline(data_path="non_existent_file.csv", model_dir=str(tmp_path))
        pipeline.load_data()

def test_column_rename(mock_pycaret_and_save, sample_data, tmp_path):
    """Stellt sicher, dass die Spalte 'duration_[days]' korrekt umbenannt wird."""
    model_dir = tmp_path / "models"
    pipeline = TrainingPipeline(data_path=sample_data, model_dir=str(model_dir))
    
    # Führe nur die Lade- und Setup-Schritte aus
    pipeline.load_data()
    pipeline.run_pycaret_setup()
    
    # Hole das DataFrame, das an pycaret.setup übergeben wurde
    # setup() wird mit kwargs aufgerufen, daher data=...
    call_args = mock_pycaret_and_save['setup'].call_args
    passed_df = call_args.kwargs['data']
    
    # Überprüfe die Spalten des übergebenen DataFrames
    assert 'duration_days' in passed_df.columns
    assert 'duration_[days]' not in passed_df.columns 

    # NEU: Prüfe, ob SMOTE aktiviert ist
    assert call_args.kwargs.get('fix_imbalance', False) is True
    assert call_args.kwargs.get('fix_imbalance_method', None) == 'smote' 

def test_pycaret_setup_smote_balancing(mock_pycaret_and_save, tmp_path):
    """Testet explizit, ob das PyCaret-Setup mit SMOTE für Balancing aufgerufen wird."""
    # Simuliere ein DataFrame mit starkem Klassenungleichgewicht
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "imbalanced_data.csv"
    df = pd.DataFrame({
        'Complaint_Type': ['A'] * 95 + ['B'] * 5,  # 95% zu 5% Verteilung
        'duration_[days]': [1.0] * 100,
        'Feature1': list(range(100)),
        'Feature2': ['X'] * 100
    })
    df.to_csv(data_file, index=False)

    model_dir = tmp_path / "models"
    pipeline = TrainingPipeline(data_path=str(data_file), model_dir=str(model_dir))
    pipeline.load_data()
    pipeline.run_pycaret_setup()

    call_args = mock_pycaret_and_save['setup'].call_args
    # Prüfe, ob Balancing aktiviert ist
    assert call_args.kwargs.get('fix_imbalance', False) is True
    assert call_args.kwargs.get('fix_imbalance_method', None) == 'smote'
    # Prüfe, ob das DataFrame ein Klassenungleichgewicht aufweist
    passed_df = call_args.kwargs['data']
    class_counts = passed_df['Complaint_Type'].value_counts()
    assert class_counts.min() / class_counts.max() < 0.2  # Mindestens 5x Unterschied 