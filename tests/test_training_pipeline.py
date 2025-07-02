import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trainAndTuneModell_fv import ModelTrainer
import pytest
from unittest.mock import MagicMock

# Minimaltest: Instanziierung
def test_modeltrainer_init():
    trainer = ModelTrainer()
    assert trainer is not None

# Mock-Test für train_models
def test_modeltrainer_train_models(monkeypatch):
    mock_train_models = MagicMock(return_value=None)
    monkeypatch.setattr(ModelTrainer, "train_models", mock_train_models)
    trainer = ModelTrainer()
    trainer.train_models(train_only_first=True)
    mock_train_models.assert_called_once_with(train_only_first=True)

# Mock-Test für evaluate_models
def test_modeltrainer_evaluate_models(monkeypatch):
    mock_evaluate = MagicMock(return_value={"xgboost": {"accuracy": 0.9}})
    monkeypatch.setattr(ModelTrainer, "evaluate_models", mock_evaluate)
    trainer = ModelTrainer()
    result = trainer.evaluate_models()
    assert result["xgboost"]["accuracy"] == 0.9
    mock_evaluate.assert_called_once()

# Mock-Test für save_model
def test_modeltrainer_save_model(monkeypatch):
    mock_save = MagicMock(return_value="/tmp/model.pkl")
    monkeypatch.setattr(ModelTrainer, "save_model", mock_save)
    trainer = ModelTrainer()
    path = trainer.save_model(model_name="xgboost")
    assert path == "/tmp/model.pkl"
    mock_save.assert_called_once_with(model_name="xgboost")

# Mock-Test für das models-Attribut
def test_modeltrainer_models_attribute():
    trainer = ModelTrainer()
    trainer.models = {"xgboost": MagicMock()}
    assert "xgboost" in trainer.models