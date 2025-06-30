"""
Fairness Audit für ein Klassifikationsmodell auf NYC 311 Complaint Daten

Dieses Skript lädt ein trainiertes Klassifikationsmodell und evaluiert dessen Fairness anhand von drei Paritätsmetriken:
- Demographic Parity Difference
- Equalized Odds Difference
- Disparate Impact Ratio

Falls eine der Metriken außerhalb eines akzeptablen Bereichs (20% Toleranz) liegt, werden automatisch Abhilfemaßnahmen durchgeführt:
1. Reweighing: Die Trainingsdaten werden gewichtet, ein neues Modell wird trainiert und erneut evaluiert.
2. Threshold Moving: Die Entscheidungsschwelle wird mit Hilfe des ThresholdOptimizers angepasst und die Fairness erneut geprüft.

Verwendete Bibliotheken:
- fairlearn: Für Fairnessmetriken und -methoden
- scikit-learn: Für das Modelltraining
- pandas: Für Datenmanipulation

Autor: Bettina Gertjerenken, Dagmar Wesemann, Kai W. Steffen
Datum: 29.06.2025
"""

#complaint_classifier_oop_20250629_153232.pkl
import joblib
import pandas as pd
import fairlearn
from colorama import Fore, Style, init
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate 
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference
#from fairlearn.metrics import disparate_impact_ratio as fairlearn_dir
from fairness_audit_ext import (
    add_reweighing_weights,
    disparate_impact_ratio,
    evaluate_fairness_per_class
)
from xgboost import XGBClassifier
from fairlearn.postprocessing import ThresholdOptimizer
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import re
from typing import Tuple, Optional, Union


def load_data(train_path, test_path):
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    for df in [X_train, X_test]:
        df.rename(columns={"duration_[days]": "duration_days"}, inplace=True)

    y_train = X_train["Complaint_Type"].values.ravel()
    y_test = X_test["Complaint_Type"].values.ravel()

    sensitive_train = X_train[["Weisse", "Afroamerikaner", "Asiaten", "Hispanics"]]
    sensitive_test = X_test[["Weisse", "Afroamerikaner", "Asiaten", "Hispanics"]]

    X_train = X_train.drop(columns=["Complaint_Type"])
    X_test = X_test.drop(columns=["Complaint_Type"])

    return X_train, X_test, y_train, y_test, sensitive_train, sensitive_test

def train_model(X_train, y_train):
    model = XGBClassifier(eval_metric="mlogloss")
    model.fit(X_train, y_train)
    return model

def evaluate_fairness(model, X_test, y_test, sensitive_test):
    classes = sorted(set(y_test))
    dpd_values, eod_values, dir_values = [], [], []

    for cls in classes:
        y_test_bin = (y_test == cls).astype(int)
        y_pred_bin = (model.predict(X_test) == cls).astype(int)

        try:
            dpd = demographic_parity_difference(y_test_bin, y_pred_bin, sensitive_features=sensitive_test)
            eod = equalized_odds_difference(y_test_bin, y_pred_bin, sensitive_features=sensitive_test)
            dir_ = disparate_impact_ratio(y_test_bin, y_pred_bin, sensitive_features=sensitive_test)
        except Exception as e:
            print(f"❌ Fehler bei Klasse {cls}: {e}")
            dpd, eod, dir_ = np.nan, np.nan, np.nan

        dpd_values.append(dpd)
        eod_values.append(eod)
        dir_values.append(dir_)

    return dpd_values, eod_values, dir_values, classes

def check_fairness_violations(dpd_values, eod_values, dir_values):
    return (
        max(dpd_values) > 0.2 or
        max(eod_values) > 0.2 or
        any((d < 0.8 or d > 1.2) for d in dir_values if not pd.isna(d))
    )

def apply_reweighing(X_train, y_train, sensitive_train):
    # 📦 Reweighing: Gewichte berechnen
    weights = add_reweighing_weights(X_train, y_train, sensitive_train)

    # 🌲 XGBoost-Modell trainieren mit sample_weight
    model_rw = XGBClassifier(
        objective="multi:softprob",        # für Multiclass
        num_class=len(set(y_train)),       # Anzahl Klassen
        eval_metric="mlogloss"
        #use_label_encoder=False
    )
    model_rw.fit(X_train, y_train, sample_weight=weights)
    return model_rw




def apply_threshold_moving(model, X_test, y_test, sensitive_test, target_class):
    y_test_bin = (y_test == target_class).astype(int)

    threshold_opt = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        prefit=True,
        predict_method="predict_proba"
    )

    threshold_opt.fit(X_test, y_test_bin, sensitive_features=sensitive_test)
    y_pred_tm = threshold_opt.predict(X_test, sensitive_features=sensitive_test)
    return y_pred_tm
