# tests/test_fairlearn_audit.py
import pytest
import pandas as pd
from xgboost import XGBClassifier
from fairness_audit import (
    load_data,
    evaluate_fairness,
    check_fairness_violations,
    apply_reweighing,
    apply_threshold_moving
)
from colorama import Fore, Style, init
init(autoreset=True)

def banner(text, color=Fore.CYAN):
    print("\n" + color + Style.BRIGHT + f"🔷 {text}")

@pytest.fixture
def data_bundle():
    return load_data(
        "data/training/train_data_fv_20250629_185546.csv",
        "data/training/test_data_oop.csv"
    )

def test_load_data_shapes(data_bundle):
    X_train, X_test, y_train, y_test, s_train, s_test = data_bundle
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert s_test.shape[0] == y_test.shape[0]
    assert "Weisse" in s_test.columns

def test_evaluate_fairness_runs(data_bundle):
    X_train, X_test, y_train, y_test, _, s_test = data_bundle
    model = XGBClassifier()
    model.fit(X_train, y_train)
    dpd, eod, dir_, classes = evaluate_fairness(model, X_test, y_test, s_test)
    assert isinstance(dpd, list)
    assert len(classes) > 0
    assert len(dpd) == len(classes)

def test_threshold_moving_for_single_class(data_bundle):
    X_train, X_test, y_train, y_test, s_train, s_test = data_bundle
    model = XGBClassifier()
    model.fit(X_train, y_train)
    target_class = 3
    y_pred = apply_threshold_moving(model, X_test, y_test, s_test, target_class)
    assert len(y_pred) == X_test.shape[0]
