"""
🧮 Fairness-Metriken für Klassifikationsmodelle – Analyse und Korrektur

Dieses Modul stellt Funktionen zur Verfügung, um Fairnessmetriken in Klassifikationsmodellen
zu berechnen und durch Reweighing gewichtet auf Fairnessverstöße zu reagieren.

Module-Funktionalitäten:
------------------------
1. disparate_impact_ratio:
   - Berechnet das Disparate Impact Ratio (DIR) für binäre Klassifikationen.
   - Gibt das Verhältnis der positiven Vorhersageraten zwischen benachteiligten und bevorzugten Gruppen an.

2. evaluate_fairness_per_class:
   - Bewertet für jede Zielklasse separat:
     • Demographic Parity Difference (DPD)
     • Equalized Odds Difference (EOD)
     • Disparate Impact Ratio (DIR)
   - Gibt eine Liste von Metriken je Klasse zurück.

3. add_reweighing_weights:
   - Berechnet für jedes Beispiel Reweighing-Gewichte basierend auf der Verteilung der Zielwerte
     innerhalb der sensitiven Gruppen.
   - Unterstützt so Fairnessmaßnahmen durch Gewichtung beim Training.

Verwendete Bibliotheken:
------------------------
- pandas: Datenmanipulation und Gruppierungen
- fairlearn.metrics: Berechnung gängiger Fairnessmetriken (DPD, EOD)

Parameter:
----------
- y_true: Echte Klassenlabels
- y_pred: Modellvorhersagen
- sensitive_features: Sensitive Gruppierungsmerkmale (z. B. Geschlecht, Ethnie)
- classes (optional): Konkrete Zielklassen zur Fairnessbewertung

Anwendungsbeispiel:
-------------------
Zur Integration in Fairnesspipelines für Machine-Learning-Projekte,
insbesondere zur Validierung von Klassifikationsmodellen im Hinblick
auf unterschiedliche Behandlung zwischen Gruppen.

Autor: Bettina Gertjerenken, Dagmar Wesemann, Kai W. Steffen
Stand: Juni 2025
"""

import pandas as pd
from fairlearn.metrics import demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference

# Berechnet das Disparate Impact Ratio für binäre Klassifikationsergebnisse
# y_true: Wahre Labels
# y_pred: Vorhergesagte Labels
# sensitive_features: Sensitive Merkmale (z.B. Geschlecht, Ethnie), als DataFrame oder Series
# Rückgabe: Verhältnis der minimalen zur maximalen positiven Rate zwischen den Gruppen

def disparate_impact_ratio(y_true, y_pred, sensitive_features):
    # Wenn sensitive_features ein DataFrame ist, verwende die erste Spalte zur Gruppierung
    if isinstance(sensitive_features, pd.DataFrame):
        group_col = sensitive_features.columns[0]
        groups = sensitive_features[group_col]
    else:
        groups = sensitive_features

    rates = []
    for group in pd.unique(groups):
        mask = (groups == group)
        if mask.sum() == 0:
            continue  # Überspringe Gruppen ohne Einträge
        # Berechne die Rate der positiven Vorhersagen (Label == 1) für die Gruppe
        rate = (y_pred[mask] == 1).mean()  # für binäre Labels: 1 = positive Klasse
        rates.append(rate)
    if len(rates) == 0 or max(rates) == 0:
        return float('nan')  # Vermeide Division durch Null
    return min(rates) / max(rates)

# Bewertet Fairness-Metriken für jede Klasse einzeln
# y_true: Wahre Labels
# y_pred: Vorhergesagte Labels
# sensitive_features: Sensitive Merkmale (z.B. Geschlecht, Ethnie)
# classes: Liste der zu betrachtenden Klassen (optional)
# Rückgabe: Liste von Dictionaries mit Fairness-Metriken pro Klasse

def evaluate_fairness_per_class(y_true, y_pred, sensitive_features, classes=None):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    sensitive_features = pd.Series(sensitive_features)
    
    results = []
    classes = sorted(set(y_true)) if classes is None else classes

    for cls in classes:
        # Binär-Kodierung für die aktuelle Klasse
        y_true_bin = (y_true == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)

        try:
            # Berechne Demographic Parity Difference
            dpd = demographic_parity_difference(y_true_bin, y_pred_bin, sensitive_features=sensitive_features)
            # Berechne Equalized Odds Difference
            eod = equalized_odds_difference(y_true_bin, y_pred_bin, sensitive_features=sensitive_features)
            # Berechne Disparate Impact Ratio
            dir_ = disparate_impact_ratio(y_true_bin, y_pred_bin, sensitive_features)

            results.append({
                "Klasse": cls,
                "DemographicParityDifference": round(float(dpd), 3),
                "EqualizedOddsDifference": round(float(eod), 3),
                "DisparateImpactRatio": round(float(dir_), 3) if not pd.isna(dir_) else "NaN"
            })
        except Exception as e:
            # Fehlerbehandlung für den Fall, dass eine Metrik nicht berechnet werden kann
            results.append({
                "Klasse": cls,
                "Fehler": str(e)
            })
    return results

# Fügt Reweighting-Gewichte für Fairness-Algorithmen hinzu
# X: Merkmalsmatrix (wird nicht verwendet, aber für API-Kompatibilität)
# y: Ziel-Labels
# sensitive_features: Sensitive Merkmale (z.B. Geschlecht, Ethnie)
# Rückgabe: Array mit Gewichten für jedes Beispiel, um Ungleichgewichte auszugleichen

def add_reweighing_weights(X, y, sensitive_features):
    # Erzeuge Gruppenschlüssel aus den sensitiven Merkmalen
    if isinstance(sensitive_features, pd.DataFrame):
        group = sensitive_features.apply(lambda row: tuple(row), axis=1)
    else:
        group = sensitive_features

    df = pd.DataFrame({'group': group, 'label': y})
    # Berechne die Häufigkeit jeder (Gruppe, Label)-Kombination
    freq = df.value_counts(normalize=True)
    # Weise jedem Beispiel das inverse der Häufigkeit als Gewicht zu
    weights = df.apply(lambda row: 1.0 / freq[(row['group'], row['label'])], axis=1)
    return weights.values
