
# 🗽 New York City 311 – FastAPI-Modellprojekt

Dieses Projekt enthält ein trainiertes Modell zur Verarbeitung von 311-Complaint-Daten aus NYC, eine FastAPI-Schnittstelle sowie diverse Prüf- und Dashboard-Komponenten zur Fairness-Analyse sowie alle zugehörigen Skripte zur Datenvorverabeitung und zum Training.
Der Originaldatensatz unter https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data ist 22 GB groß. Die Modellstruktur ist nun so aufgebaut, dass sich nach der Vorverarbeitung ein sinnvoller Einstiegspunkt 
für das Training und Tuning der Modelle auf Basis hinterlegter reduzierter Datensätze ergibt.

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
## 🚀 Schnellstar

### 1. Datenvorverarbeitung mit 
```bash
Daten_vorbereiten.py
```
und 
```bash
data_preprocessing.py
```
liefert bzgl. "Complaint Type" stratifizierte csv Dateien mit Trainings- und Testdatensätzen, mit sklearn Transformern Datum-Zu-Feature, High-Cardinality-Encoding und Outlier-Handling für
das weitere Training vorbereitet. 

### 2. Training mit 
```bash
trainAndTuneModell_fv.py
```
Kann mit verschiedenen Optionen, z.B. --train-only-first --skip-tuning aufgerufen werden.

### 3. Fast api lokal starten aus dem Hauptprojetkverzeichnis mit

```bash
uvicorn src.main_api_final:app --reload
```

→ API läuft dann unter: [http://localhost:8000/docs](http://localhost:8000/docs)


### 4. Streamlit-Fairness-Dashboard starten

```bash
streamlit run src/Fairness_dashboard.py
```

→ Dashboard öffnet sich automatisch im Browser (Port 8501)

---

### 3. Tests ausführen

Einzelne Tests z.B. mit
```bash
pytest tests/test_main_api_fv.py
```

Für einen vollständigen Testlauf für die gesamte Pipeline:

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

