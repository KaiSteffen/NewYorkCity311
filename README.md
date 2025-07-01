
# 🗽 New York City 311 – FastAPI-Modellprojekt

Dieses Projekt enthält ein trainiertes Modell zur Verarbeitung von 311-Complaint-Daten aus NYC, eine FastAPI-Schnittstelle sowie diverse Prüf- und Dashboard-Komponenten zur Fairness-Analyse.

---

## 📁 Projektstruktur (Auszug)

```
├── src/
│   └── main.py                 → Startpunkt für die FastAPI-Anwendung
│   └── model_utils.py          → Funktionen zum Laden & Vorverarbeiten des Modells
│   └── call_fairness_audit.py  → Funktion zum Aufruf der Fairness-Analyse
│   └── data_processing.py      → Funktion zur Datenanalyse und Vorverarbeitung
│   └── generate_test_data_final.py → Generierung von Testdaten im json -Format│
├── data/                       → Trainings-/Testdaten
│
├── tests/
│   └── test_main_api_fv.py     → Tests für API-Endpunkte, Datenmodelle & Integration
│   └── test_fairlearn_audit.py → Fairness-Test über `fairlearn`
│   └── test_dashboard.py       → Test für das Fairness-Dashboard
│
├── models/
│   └── complaint_classifier.pkl → Hauptmodell nach PyCaret-Tuning
│   └── reweighed_model.pkl      → Debiased-Modell (Fairness-korrigiert)
│   └── threshold_preds.npy      → Schwellen-angepasste Vorhersagen
│
├── Fairness_dashboard.py       → Streamlit-Dashboard zur Modellvisualisierung
├── README.md
```

---

## 🚀 Schnellstart

### 1. FastAPI lokal starten

```bash
uvicorn src.main:app --reload
```

→ API läuft dann unter: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Streamlit-Fairness-Dashboard starten

```bash
streamlit run src/Fairness_dashboard.py
```

→ Dashboard öffnet sich automatisch im Browser (Port 8501)

---

### 3. Tests ausführen

```bash
pytest tests/test_main_api_fv.py
```

Für einen vollständigen Testlauf:

```bash
pytest tests/
```

---

### 4. Fairness-Audit separat durchführen

```bash
pytest tests/test_fairlearn_audit.py
```

Alternativ kannst du den Audit auch direkt als Notebook oder Dashboard ausführen.

---

## 🧠 Hinweise

- Modelle liegen im `models/`-Ordner, inkl. versionierter Exportdateien
- Projektstruktur ist `src`-basiert, für klare Modularisierung
- `.gitkeep`-Dateien dienen nur dazu, leere Ordner in Git zu speichern

---

📝 Bei Fragen oder Beiträgen, einfach Issues oder Pull Requests eröffnen. Viel Spaß mit dem Projekt!
```

# 🗽 New York City 311 – FastAPI-Modellprojekt

Dieses Projekt enthält ein trainiertes Modell zur Verarbeitung von 311-Complaint-Daten aus NYC, eine FastAPI-Schnittstelle sowie diverse Prüf- und Dashboard-Komponenten zur Fairness-Analyse.

---

## 📁 Projektstruktur (Auszug)

