# tests/test_fairness_analysis_flow.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import joblib
from xgboost import XGBClassifier
import os
from fairness_audit import (
    load_data,
    evaluate_fairness,
    check_fairness_violations,
    apply_reweighing,
    apply_threshold_moving
)
from colorama import Fore, Style, init
import pandas as pd

init(autoreset=True)

def banner(text, color=Fore.CYAN):
    print("\n" + color + Style.BRIGHT + f"🔷 {text}")

@pytest.fixture
def data_bundle():
    return load_data("data/train_data_final.csv", "data/test_data_final.csv")

@pytest.fixture
def fairness_df():
    df = pd.read_csv("results/fairness_metrics.csv")
    return df

def test_full_fairness_analysis(data_bundle):
    banner("🔍 Starte Fairnessanalyse-Ende-zu-Ende-Test", Fore.GREEN)

    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = data_bundle

    banner("📦 Lade & überprüfe Modell", Fore.BLUE)
    model = XGBClassifier()
    model.load_model("models/model_reexported_20250630_124707.json")
    assert model is not None
    print("✅ Modell erfolgreich geladen.")

    banner("📊 Berechne Fairnessmetriken", Fore.YELLOW)
    dpd, eod, dir_, classes = evaluate_fairness(model, X_test, y_test, sensitive_test)
    print(f"🔢 Klassenanzahl: {len(classes)}")
    print(f"📐 Max. DPD: {max(dpd):.3f}, Max. EOD: {max(eod):.3f}, Min. DIR: {min(dir_):.3f}")
    assert len(dpd) == len(classes)

    if check_fairness_violations(dpd, eod, dir_):
        banner("🚨 Fairnessverletzung erkannt – teste Reweighing + Threshold-Moving", Fore.RED)

        print("♻️ Reweighing...")
        reweighted_model = apply_reweighing(X_train, y_train, sensitive_train)
        assert reweighted_model is not None
        print("✅ Reweighing erfolgreich.")

        print("🎯 Threshold-Moving auf Klasse 3...")
        y_pred_tm = apply_threshold_moving(reweighted_model, X_test, y_test, sensitive_test, target_class=3)
        assert len(y_pred_tm) == len(y_test)

        joblib.dump(y_pred_tm, "models/test_thresholdmoved_preds_target3.npy")
        print("✅ Threshold-Moving abgeschlossen und Vorhersagen gespeichert.")
    else:
        print(Fore.GREEN + "✅ Keine Fairnessverletzungen – keine Korrekturen notwendig.")
