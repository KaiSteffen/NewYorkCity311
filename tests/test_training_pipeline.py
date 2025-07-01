import sys
from unittest.mock import MagicMock
import numpy as np

# PyCaret und Untermodul global mocken, bevor ModelTrainer importiert wird
mock_pycaret = MagicMock()
sys.modules['pycaret'] = MagicMock()
sys.modules['pycaret.classification'] = mock_pycaret

import pytest
import pandas as pd
from pathlib import Path
import os

# Füge das 'src'-Verzeichnis zum Python-Pfad hinzu
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Importiere die zu testende Klasse
from trainAndTuneModell_fv import ModelTrainer

@pytest.fixture
def mock_pycaret_and_save(monkeypatch):
    # Dummy-Modell und Dummy-DataFrame für die PyCaret-Mocks
    dummy_model = MagicMock()
    def make_dummy_predictions(*args, **kwargs):
        data = kwargs.get('data', None)
        if data is None and len(args) > 1:
            data = args[1]
        n = len(data) if data is not None else 10
        # Versuche, alle Klassen aus der echten Spalte zu nehmen
        if data is not None and 'Complaint_Type' in data.columns:
            unique_classes = data['Complaint_Type'].unique()
            return pd.DataFrame({'prediction_label': np.random.choice(unique_classes, size=n)})
        # Fallback: 10 Klassen
        return pd.DataFrame({'prediction_label': np.random.choice(range(10), size=n)})

    # Mock für PyCaret-Funktionen
    mock_setup = MagicMock()
    mock_compare = MagicMock()
    mock_tune = MagicMock()
    mock_finalize = MagicMock(return_value=dummy_model)
    mock_save = MagicMock()
    mock_create_model = MagicMock(return_value=dummy_model)
    mock_predict_model = MagicMock(side_effect=make_dummy_predictions)
    mock_plot_model = MagicMock()
    mock_pull = MagicMock()

    mock_pycaret.setup = mock_setup
    mock_pycaret.compare_models = mock_compare
    mock_pycaret.tune_model = mock_tune
    mock_pycaret.finalize_model = mock_finalize
    mock_pycaret.save_model = mock_save
    mock_pycaret.create_model = mock_create_model
    mock_pycaret.predict_model = mock_predict_model
    mock_pycaret.plot_model = mock_plot_model
    mock_pycaret.pull = mock_pull

    import trainAndTuneModell_fv
    trainAndTuneModell_fv.__dict__['setup'] = mock_setup
    trainAndTuneModell_fv.__dict__['compare_models'] = mock_compare
    trainAndTuneModell_fv.__dict__['tune_model'] = mock_tune
    trainAndTuneModell_fv.__dict__['finalize_model'] = mock_finalize
    trainAndTuneModell_fv.__dict__['save_model'] = mock_save
    trainAndTuneModell_fv.__dict__['create_model'] = mock_create_model
    trainAndTuneModell_fv.__dict__['predict_model'] = mock_predict_model
    trainAndTuneModell_fv.__dict__['plot_model'] = mock_plot_model
    trainAndTuneModell_fv.__dict__['pull'] = mock_pull

    # Mock für Speicherfunktionen
    mock_joblib_dump = MagicMock()
    monkeypatch.setattr('trainAndTuneModell_fv.joblib.dump', mock_joblib_dump)

    return {
        'setup': mock_setup,
        'compare_models': mock_compare,
        'tune_model': mock_tune,
        'finalize_model': mock_finalize,
        'save_model': mock_save,
        'create_model': mock_create_model,
        'predict_model': mock_predict_model,
        'plot_model': mock_plot_model,
        'pull': mock_pull,
        'joblib_dump': mock_joblib_dump
    }

@pytest.fixture
def sample_data(tmp_path):
    """Erstellt eine temporäre CSV-Datei mit Beispieldaten."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "sample_data.csv"
    df = pd.DataFrame({
        'Complaint_Type': ['HEAT/HOT WATER', 'Noise', 'HEAT/HOT WATER'],
        'duration_[days]': [1.0, 2.5, 0.5],
        'Feature1': [10, 20, 30],
        'Feature2': ['A', 'B', 'A']
    })
    df.to_csv(data_file, index=False)
    return str(data_file)

def test_pipeline_runs_without_tuning(mock_pycaret_and_save, sample_data, tmp_path):
    model_dir = tmp_path / "models"
    pipeline = ModelTrainer(data_path=sample_data, model_save_path=str(model_dir))
    pipeline.run_training_pipeline(skip_tuning=True)
    mock_pycaret_and_save['setup'].assert_called_once()
    assert mock_pycaret_and_save['create_model'].call_count > 0
    mock_pycaret_and_save['joblib_dump'].assert_called_once()
    mock_pycaret_and_save['tune_model'].assert_not_called()

def test_pipeline_runs_with_tuning(mock_pycaret_and_save, sample_data, tmp_path):
    model_dir = tmp_path / "models"
    pipeline = ModelTrainer(data_path=sample_data, model_save_path=str(model_dir))
    pipeline.run_training_pipeline(skip_tuning=False)
    mock_pycaret_and_save['setup'].assert_called_once()
    assert mock_pycaret_and_save['create_model'].call_count > 0
    assert mock_pycaret_and_save['joblib_dump'].call_count >= 1
    assert mock_pycaret_and_save['tune_model'].call_count >= 1

def test_data_loading_error(tmp_path, monkeypatch):
    """Testet, ob ein FileNotFoundError ausgelöst wird, wenn die Datendatei nicht existiert."""
    # Patche pandas.read_csv, damit garantiert ein FileNotFoundError geworfen wird
    monkeypatch.setattr('pandas.read_csv', lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(FileNotFoundError):
        pipeline = ModelTrainer(data_path="non_existent_file.csv", model_save_path=str(tmp_path))
        pipeline.load_and_split_data()

def test_column_rename(mock_pycaret_and_save, sample_data, tmp_path):
    model_dir = tmp_path / "models"
    pipeline = ModelTrainer(data_path=sample_data, model_save_path=str(model_dir))
    pipeline.load_and_split_data()
    pipeline.setup_pycaret()
    call_args = mock_pycaret_and_save['setup'].call_args
    passed_df = call_args.kwargs['data']
    assert 'duration_days' in passed_df.columns or 'duration_[days]' in passed_df.columns
    assert call_args.kwargs.get('fix_imbalance', False) is True
    assert call_args.kwargs.get('fix_imbalance_method', None) == 'smote'

def test_pycaret_setup_smote_balancing(mock_pycaret_and_save, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_file = data_dir / "imbalanced_data.csv"
    df = pd.DataFrame({
        'Complaint_Type': ['A'] * 95 + ['B'] * 5,
        'duration_[days]': [1.0] * 100,
        'Feature1': list(range(100)),
        'Feature2': ['X'] * 100
    })
    df.to_csv(data_file, index=False)
    model_dir = tmp_path / "models"
    pipeline = ModelTrainer(data_path=str(data_file), model_save_path=str(model_dir))
    pipeline.load_and_split_data()
    pipeline.setup_pycaret()
    call_args = mock_pycaret_and_save['setup'].call_args
    assert call_args.kwargs.get('fix_imbalance', False) is True
    assert call_args.kwargs.get('fix_imbalance_method', None) == 'smote'
    passed_df = call_args.kwargs['data']
    class_counts = passed_df['Complaint_Type'].value_counts()
    assert class_counts.min() / class_counts.max() < 0.2

def test_pipeline_runs_xgboost_only_no_tuning(mock_pycaret_and_save, sample_data, tmp_path):
    model_dir = tmp_path / "models"
    pipeline = ModelTrainer(data_path=sample_data, model_save_path=str(model_dir))
    pipeline.run_training_pipeline(train_only_first=True, skip_tuning=True)
    mock_pycaret_and_save['setup'].assert_called_once()
    assert mock_pycaret_and_save['create_model'].call_count > 0
    mock_pycaret_and_save['joblib_dump'].assert_called_once()
    mock_pycaret_and_save['tune_model'].assert_not_called() 