import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Daten laden: Fairness-Metriken aus CSV-Datei einlesen
# Erwartet wird eine Datei mit Metriken wie DPD, EOD, DIR pro Klasse

df = pd.read_csv("results/fairness_metrics.csv")

# 🎯 Titel und Beschreibung des Dashboards
st.title("📊 Fairness-Dashboard: Complaint Classifier")
st.markdown("""
Dieses Dashboard visualisiert die Fairnessmetriken für verschiedene Klassen im NYC 311 Klassifikationsmodell.
- **DPD > 0.2** → potenziell unfair (Demographic Parity)
- **EOD > 0.2** → potenziell unfair (Equalized Odds)
- **DIR < 0.8 oder > 1.2** → potenziell unfair (Disparate Impact)
""")

# ✅ Auswahloptionen für die zu visualisierende Fairnessmetrik und den Schwellenwert
metric = st.selectbox("Wähle Fairnessmetrik", ["DemographicParityDifference", "EqualizedOddsDifference", "DisparateImpactRatio"])
threshold = st.slider("Schwellwert zur Markierung (nur visuell)", 0.0, 1.5, 0.2, step=0.05)

# 🧼 Umwandeln: Werte der gewählten Metrik in numerisches Format bringen
# Fehlerhafte Werte werden zu NaN

df[metric] = pd.to_numeric(df[metric], errors="coerce")

# 📊 Plot: Balkendiagramm der Fairnessmetrik pro Klasse
fig, ax = plt.subplots(figsize=(10, 5))
# Farben: Rot für potenziell unfaire Werte, sonst grün
colors = ["red" if (
    (metric in ["DemographicParityDifference", "EqualizedOddsDifference"] and val > threshold) or
    (metric == "DisparateImpactRatio" and (val < 0.8 or val > 1.2))
) else "green" for val in df[metric]]

bars = plt.bar(df["Klasse"].astype(str), df[metric], color=colors)
plt.axhline(y=threshold, color='gray', linestyle='--', label=f"Schwellwert {threshold}")
plt.title(f"Fairness: {metric}")
plt.xlabel("Klasse")
plt.ylabel(metric)
plt.xticks(rotation=45)
plt.legend()

# 📈 Werte auf den Balken anzeigen
for bar, val in zip(bars, df[metric]):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{val:.2f}", ha='center', va='bottom')

# Plot im Streamlit-Dashboard anzeigen
st.pyplot(fig)

# 📥 Download-Option für die Fairness-Metriken als CSV-Datei
st.download_button("📥 Fairness-Metriken als CSV", data=df.to_csv(index=False), file_name="fairness_metrics.csv")

