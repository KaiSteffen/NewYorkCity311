"""
📊 Fairness-Dashboard für den NYC 311 Complaint Classifier
===========================================================

Dieses Streamlit-Dashboard visualisiert Fairnessmetriken wie:
- Demographic Parity Difference (DPD)
- Equalized Odds Difference (EOD)
- Disparate Impact Ratio (DIR)

Ziel ist es, potenzielle Verzerrungen im Klassifikationsmodell anhand dieser Metriken sichtbar zu machen.

Autor: [Dein Name]
Letzte Änderung: [Datum]
"""

# 📚 Bibliotheken importieren
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔠 Mapping-Funktionen für sprechende Bezeichnungen
CLASS_MAPPING = {
    0: 'Noise - Residential', 1: 'HEAT/HOT WATER', 2: 'Blocked Driveway', 3: 'Illegal Parking',
    4: 'Street Condition', 5: 'Street Light Condition', 6: 'Water System', 7: 'UNSANITARY CONDITION',
    8: 'Noise - Commercial', 9: 'PLUMBING', 10: 'Noise', 11: 'Derelict Vehicles', 12: 'Rodent',
    13: 'Noise - Street/Sidewalk', 14: 'Damaged Tree', 15: 'Missed Collection (All Materials)',
    16: 'General Construction/Plumbing', 17: 'Noise - Vehicle', 18: 'Taxi Complaint',
    19: 'Consumer Complaint', 20: 'Root/Sewer/Sidewalk Condition', -1: 'Unknown Class'
}

OUTPUT_GROUPING_MAPPING = {
    'Noise - Residential': 'Lärmbelästigung', 'Noise - Commercial': 'Lärmbelästigung',
    'Noise': 'Lärmbelästigung', 'Noise - Street/Sidewalk': 'Lärmbelästigung',
    'Noise - Vehicle': 'Lärmbelästigung', 'Blocked Driveway': 'Verkehrsprobleme',
    'Illegal Parking': 'Verkehrsprobleme', 'Derelict Vehicles': 'Verkehrsprobleme',
    'Street Condition': 'Zustand öffentlicher Raum', 'Street Light Condition': 'Zustand öffentlicher Raum',
    'Damaged Tree': 'Zustand öffentlicher Raum', 'Root/Sewer/Sidewalk Condition': 'Zustand öffentlicher Raum',
    'HEAT/HOT WATER': 'Gebäude & Versorgung', 'PLUMBING': 'Gebäude & Versorgung',
    'Water System': 'Gebäude & Versorgung', 'General Construction/Plumbing': 'Gebäude & Versorgung',
    'UNSANITARY CONDITION': 'Sauberkeit & Hygiene', 'Rodent': 'Sauberkeit & Hygiene',
    'Missed Collection (All Materials)': 'Sauberkeit & Hygiene', 'Taxi Complaint': 'Sonstige Dienstleistungen',
    'Consumer Complaint': 'Sonstige Dienstleistungen'
}















def get_readable_class_label(class_id):
    base = CLASS_MAPPING.get(class_id, 'Unbekannte Klasse')
    group = OUTPUT_GROUPING_MAPPING.get(base, 'Sonstiges')
    return f"{group} ({base})"

# 💡 Alle möglichen Klassennamen zur Auswahl vorbereiten
alle_klassen = sorted([get_readable_class_label(k) for k in CLASS_MAPPING.keys()])


# 🔄 Standarddesign für Diagramme setzen
sns.set(style="whitegrid")

# 📥 Fairness-Metriken laden
# Erwartet wird eine CSV-Datei im Verzeichnis results/ mit folgenden Spaltennamen:
# - DemographicParityDifference
# - EqualizedOddsDifference
# - DisparateImpactRatio
try:
    df = pd.read_csv("results/fairness_metrics.csv")
except FileNotFoundError:
    st.error("❌ Die Datei 'results/fairness_metrics.csv' konnte nicht gefunden werden.")
    st.stop()

# 🧾 Titel und Dashboard-Beschreibung
st.title("📊 Fairness-Dashboard: Complaint Classifier")
st.markdown(
    """
    Dieses Dashboard zeigt Fairnessmetriken für verschiedene Klassen im NYC 311 Klassifikationsmodell.

    **Interpretation der Metriken:**
    - **DPD > 0.2** → potenziell unfair (Demographic Parity)
    - **EOD > 0.2** → potenziell unfair (Equalized Odds)
    - **DIR < 0.8 oder > 1.2** → potenziell unfair (Disparate Impact)
    """
)

# 🎛 Benutzerdefinierte Auswahl der Metrik und Schwelle
metric = st.selectbox(
    "🔎 Wähle eine Fairnessmetrik zur Visualisierung:",
    ["DemographicParityDifference", "EqualizedOddsDifference", "DisparateImpactRatio"]
)

# ausgewählte_klasse = st.selectbox("🎯 Wähle eine Klasse zur Filterung (optional):", ["Alle"] + alle_klassen)

# # ✅ Gruppierte Visualisierung
# if "group" in df.columns:
#     fig_group, ax_group = plt.subplots(figsize=(10, 5))
#     sns.barplot(x="group", y=metric, data=df, palette="pastel", ax=ax_group)
#     ax_group.set_title(f"{metric} nach Bevölkerungsgruppe")
#     ax_group.axhline(threshold, color="red", linestyle="--", label=f"Schwellwert: {threshold}")
#     ax_group.set_ylabel(metric)
#     ax_group.set_xlabel("Bevölkerungsgruppe")
#     ax_group.legend()
#     st.subheader("📊 Vergleich nach Ethnischer Zugehörigkeit")
#     st.pyplot(fig_group)
# else:
#     st.info("ℹ️ Keine Gruppendaten vorhanden – gruppierte Anzeige nicht möglich.")



# 📘 Interpretation je nach ausgewählter Metrik
interpretationen = {
    "DemographicParityDifference": """
**📘 Demographic Parity Difference (DPD):**

- Misst, ob verschiedene Gruppen dieselbe Wahrscheinlichkeit haben, eine positive Vorhersage zu erhalten.
- **DPD > 0.2** deutet auf mögliche Ungleichbehandlung hin.
- Beispiel: Wenn Beschwerden einer Bevölkerungsgruppe seltener in eine vorteilhaftere Beschwerdekategorie klassifiziert werden, ist das Modell ggf. verzerrt.

➡️ Ziel: DPD möglichst nahe bei 0.
""",

    "EqualizedOddsDifference": """
**📘 Equalized Odds Difference (EOD):**

- Betrachtet Unterschiede in den Fehlerquoten (False Positives & False Negatives) zwischen Gruppen.
- **EOD > 0.2** → Das Modell macht häufiger Fehler für bestimmte Bevölkerungsgruppen.
- Das ist kritisch, wenn Beschwerden einer Ethnien-Gruppen häufiger falsch klassifiziert werden als einer anderen Gruppe.

➡️ Ziel: EOD nahe bei 0 – gleich gute Behandlung aller Gruppen.
""",

    "DisparateImpactRatio": """
**📘 Disparate Impact Ratio (DIR):**

- Verhältnis positiver Vorhersagen zwischen Bevölkerungsgruppen.
- **DIR < 0.8 oder > 1.2** → Hinweis auf unfaire Bevorzugung oder Benachteiligung.
- Beispiel: Wenn Menschen einer bestimmten Bevölkerungsgruppe/Ethnie seltener eine positive Klassifikation bekommen.

➡️ Ziel: DIR zwischen 0.8 und 1.2 für faire Ergebnisse.
"""
}

# 📘 Interpretation anzeigen
st.markdown(interpretationen[metric])


threshold = st.slider(
    "📏 Visuelle Schwelle zur Markierung potenzieller Verzerrung:",
    min_value=0.0,
    max_value=1.5,
    value=0.2,
    step=0.05
)

# 🧼 Datenvorverarbeitung: ausgewählte Metrik in numerischen Typ konvertieren
df[metric] = pd.to_numeric(df[metric], errors="coerce")

# Hinweis bei fehlenden oder unlesbaren Werten
if df[metric].isna().all():
    st.warning(f"⚠️ Keine gültigen Werte für die Metrik '{metric}' vorhanden.")
    st.stop()

# 📊 Balkendiagramm anzeigen
st.subheader(f"📈 Visualisierung: {metric}")
fig, ax = plt.subplots(figsize=(10, 5))
df["Label"] = df.index.map(get_readable_class_label)
#bar = sns.barplot(x=df["Label"], y=df[metric], palette="viridis", ax=ax)
bar = sns.barplot(x="Label", y=metric, data=df, palette="viridis", ax=ax)
ax.set_xlabel("Klassenbezeichnung")

# Schwellenwert visuell einzeichnen
ax.axhline(threshold, color="red", linestyle="--", label=f"Schwellwert: {threshold}")
ax.set_xlabel("Beschwerdekategorie")
ax.set_ylabel(metric)
ax.set_title(f"{metric} pro Klasse")
ax.legend()

st.pyplot(fig)

