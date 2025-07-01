"""
Dieses Skript führt eine Fairnessanalyse auf einem vortrainierten XGBoost-Modell durch
und verbessert bei Bedarf die Fairness durch Reweighing und Threshold-Moving.

Funktionen und Workflow:
------------------------
1. Laden von Trainings- und Testdaten sowie sensitiven Attributen.
2. Laden eines vortrainierten Klassifikationsmodells.
3. Berechnung von Fairnessmetriken (Demographic Parity Difference, Equal Opportunity Difference, Disparate Impact Ratio).
4. Automatisierte Überprüfung auf Fairnessverstöße.
5. Bei Verstößen:
    a) Reweighing zur Fairnesskorrektur in den Trainingsdaten.
    b) Anwendung von Threshold-Moving auf eine Zielklasse.
    c) Speicherung des angepassten Modells und der Vorhersagen.
6. Visualisierung und Logging an relevanten Stellen zur Nachvollziehbarkeit.

Verwendete Module:
------------------
- fairness_audit: eigenes Modul für Fairnessanalyse und Fairnessmaßnahmen.
- xgboost: Gradient Boosting Classifier.
- joblib: zum Speichern und Laden von Modellen und Daten.
- sklearn, numpy, seaborn, matplotlib: für Analysen und Visualisierungen.

Hinweis:
--------
- Die Zielklasse für Threshold-Moving ist aktuell auf 3 gesetzt (anpassbar).
- Modell und Datenpfade sind projektspezifisch und sollten ggf. konfigurierbar gemacht werden.

Autor: Bettina Gertjerenken, Dagmar Wesemann, Kai W. Steffen
Stand: Juni 2025
"""

# 📦 Fairness-Funktionen aus eigenem Modul importieren
from fairness_audit import (
    evaluate_fairness,
    apply_reweighing,
    apply_threshold_moving,
    check_fairness_violations,
    load_data,
    train_model
)

# ⚙️ Standardbibliotheken und Tools
import joblib
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Daten laden
print("🔄 Lade Trainings- und Testdaten...")
X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = load_data(
    "data/train_data_final.csv",
    "data/test_data_final.csv"
)
print("✅ Daten erfolgreich geladen.\n")

# 📦 Modell laden 
model_path = "models/model_reexported_20250630_124707.json"
print(f"📦 Lade bereits trainiertes Modell aus: {model_path}")
model = XGBClassifier()
model.load_model(model_path)

with open(model_path, "r") as f:
    head = f.read(100)
    print("🔍 Modellvorschau:", head[:100], "...\n")

# 📊 Fairnessanalyse
print("📊 Starte Fairnessanalyse mit dem geladenen Modell...")
dpd, eod, dir_, classes = evaluate_fairness(model, X_test, y_test, sensitive_test)
print("✅ Fairnessmetriken erfolgreich berechnet.\n")

# ➕ Ausgabe der Fairnessmetriken pro Klasse
print("--- Fairnessmetriken pro Klasse ---")
for cls, dpd_val, eod_val, dir_val in zip(classes, dpd, eod, dir_):
    print(f"Klasse: {cls}")
    print(f"  Demographic Parity Difference (DPD): {dpd_val:.3f}")
    print(f"  Equalized Odds Difference (EOD):     {eod_val:.3f}")
    print(f"  Disparate Impact Ratio (DIR):        {dir_val:.3f}\n")

# 🛡️ Fairness überprüfen
if check_fairness_violations(dpd, eod, dir_):
    print("🚨 Fairnessverletzung erkannt – starte Reweighing + Threshold-Moving.\n")

    # 🔁 Reweighing durchführen (Sample-Gewichtung basierend auf Gruppenverteilung)
    print("♻️ Führe Reweighing durch...")
    model_rw = apply_reweighing(X_train, y_train, sensitive_train)
    joblib.dump(model_rw, "models/reweighed_model.pkl")
    print("✅ Modell mit Reweighing erfolgreich trainiert und gespeichert.\n")

    # 📈 Vorhersagen des regewichteten Modells
    y_pred_rw = model_rw.predict(X_test)

    # 🎯 Threshold-Moving auf eine ausgewählte Zielklasse anwenden
    target_class = 3  # Beispiel: Klasse 3 (kann angepasst werden)
    print(f"🎯 Wende Threshold-Moving auf Zielklasse {target_class} an...")
    y_pred_tm = apply_threshold_moving(model_rw, X_test, y_test, sensitive_test, target_class)

    # 💾 Speichern der Schwellenwert-korrigierten Vorhersagen
    joblib.dump(y_pred_tm, "models/thresholdmoved_preds_target3.npy")
    print("✅ Threshold-Moving abgeschlossen und Vorhersagen gespeichert.\n")

else:
    print("✅ Alle Fairnessmetriken im akzeptablen Bereich – kein Eingreifen erforderlich.\n")

