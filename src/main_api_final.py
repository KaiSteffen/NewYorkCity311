import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Optional
import pandas as pd
from pathlib import Path
import os
import numpy as np
from contextlib import asynccontextmanager
import traceback
from enum import Enum
from datetime import datetime

# --- NEU: DateFeatureTransformer direkt in die API-Datei kopiert ---
class DateFeatureTransformer:
    """Minimaler Transformer, um die benötigten Datums-Features zu extrahieren."""
    def __init__(self, date_column_name='created_date'):
        self.date_column_name = date_column_name
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # In datetime-Objekt umwandeln
        dt_series = pd.to_datetime(df[self.date_column_name])
        
        # Features extrahieren
        df['Created_Date_hour'] = dt_series.dt.hour
        df['Created_Date_month'] = dt_series.dt.month
        df['Created_Date_dayofweek'] = dt_series.dt.dayofweek
        
        return df

# --- Globale Variablen für Mapping-Daten ---
ZIP_BOROUGH_MAPPING = {}
BOROUGH_DEMOGRAPHICS_MAPPING = {}
DEMOGRAPHICS_FALLBACK = {'Weisse': 0.0, 'Afroamerikaner': 0.0, 'Asiaten': 0.0, 'Hispanics': 0.0}

# --- Model Loading Globals ---
MODEL_DIR = Path(os.getenv("MODEL_DIR", r"D:\Fernstudium\Module\AITools\repo\311NYC\models"))
SPECIFIC_MODEL_PATH = MODEL_DIR / "model_reexported_20250630_124707.json" # Neuestes sauberes Modell
BOOSTER = None
FEATURE_NAMES = [
    'Agency', 'Location_Type', 'Incident_Zip', 'Borough', 'Vehicle_Type',
    'duration_days', 'Created_Date_hour', 'Created_Date_month',
    'Created_Date_dayofweek', 'Descriptor_encoded', 'Weisse',
    'Afroamerikaner', 'Asiaten', 'Hispanics'
]

# --- Hardcoded Mappings ---
# Diese Mappings sind für die interne Umwandlung der Daten, die das Modell erwartet
AGENCY_MAPPING = {
    'DSNY': 0, 'DOT': 1, 'DPR': 2, 'HPD': 3, 'NYPD': 4, 'DEP': 5, 'DOB': 6,
    'DOHMH': 7, 'TLC': 8, 'FDNY': 9, 'DOE': 10, 'EDC': 11, 'DOITT': 12, '3-1-1': 13
}
LOCATION_TYPE_MAPPING = {
    'Street/Sidewalk': 0, 'Residential Building/House': 1, 'Park/Playground': 2,
    'Commercial': 3, 'Store/Commercial': 4, 'Club/Bar/Restaurant': 5, 'House of Worship': 6,
    'Residential Building': 7, 'Highway': 8, 'Parking Lot': 9, 'Catch Basin': 10, 'Subway Station': 11,
    'Other': 98, 'UNKNOWN': 99
}
BOROUGH_MAPPING = {
    'MANHATTAN': 0, 'BROOKLYN': 1, 'QUEENS': 2, 'BRONX': 3, 'STATEN ISLAND': 4, 'Unspecified': 5
}
VEHICLE_TYPE_MAPPING = {
    'none': 0, 'N/A': 1, 'PASSENGER VEHICLE': 2, 'TRUCK': 3, 'MOTORCYCLE': 4,
    'BUS': 5, 'TAXI': 6, 'LIVERY VEHICLE': 7, 'VAN': 8, 'SCOOTER': 9
}
DESCRIPTOR_MAPPING = {
    'Loud Music/Party': 0, 'Noise - Street/Sidewalk': 1, 'Noise - Commercial': 2,
    'Blocked Driveway': 3, 'Illegal Parking': 4, 'Street Light Condition': 5,
    'Street Condition': 6, 'Water System': 7, 'Sewer': 8, 'HEAT/HOT WATER': 9,
    'Plumbing': 10, 'Paint/Plaster': 11, 'Derelict Vehicle': 12, 'Noise - Vehicle': 13,
    'Rodent': 14, 'UNSANITARY CONDITION': 15, 'Dirty Conditions': 16, 'Water Quality': 17,
    'Damaged Tree': 18, 'Missed Collection': 19, 'General Construction': 20,
    'FLOORING/STAIRS': 21, 'DOOR/WINDOW': 22, 'Taxi Complaint': 23, 'Consumer Complaint': 24,
    'sonstiges': 99
}
CLASS_MAPPING = {
    0: 'Noise - Residential', 1: 'HEAT/HOT WATER', 2: 'Blocked Driveway', 3: 'Illegal Parking',
    4: 'Street Condition', 5: 'Street Light Condition', 6: 'Water System', 7: 'UNSANITARY CONDITION',
    8: 'Noise - Commercial', 9: 'PLUMBING', 10: 'Noise', 11: 'Derelict Vehicles', 12: 'Rodent',
    13: 'Noise - Street/Sidewalk', 14: 'Damaged Tree', 15: 'Missed Collection (All Materials)',
    16: 'General Construction/Plumbing', 17: 'Noise - Vehicle', 18: 'Taxi Complaint',
    19: 'Consumer Complaint', 20: 'Root/Sewer/Sidewalk Condition',
    -1: 'Unknown Class'
}

# --- NEU: Finale Gruppierung für die API-AUSGABE ---
OUTPUT_GROUPING_MAPPING = {
    # Lärm
    'Noise - Residential': 'Lärmbelästigung',
    'Noise - Commercial': 'Lärmbelästigung',
    'Noise': 'Lärmbelästigung',
    'Noise - Street/Sidewalk': 'Lärmbelästigung',
    'Noise - Vehicle': 'Lärmbelästigung',
    # Verkehr
    'Blocked Driveway': 'Verkehrsprobleme',
    'Illegal Parking': 'Verkehrsprobleme',
    'Derelict Vehicles': 'Verkehrsprobleme',
    # Zustand öffentlicher Raum
    'Street Condition': 'Zustand öffentlicher Raum',
    'Street Light Condition': 'Zustand öffentlicher Raum',
    'Damaged Tree': 'Zustand öffentlicher Raum',
    'Root/Sewer/Sidewalk Condition': 'Zustand öffentlicher Raum',
    # Gebäude & Versorgung
    'HEAT/HOT WATER': 'Gebäude & Versorgung',
    'PLUMBING': 'Gebäude & Versorgung',
    'Water System': 'Gebäude & Versorgung',
    'General Construction/Plumbing': 'Gebäude & Versorgung',
    # Sauberkeit & Hygiene
    'UNSANITARY CONDITION': 'Sauberkeit & Hygiene',
    'Rodent': 'Sauberkeit & Hygiene',
    'Missed Collection (All Materials)': 'Sauberkeit & Hygiene',
    # Sonstiges
    'Taxi Complaint': 'Sonstige Dienstleistungen',
    'Consumer Complaint': 'Sonstige Dienstleistungen',
}

# --- NEU: Gruppierung für benutzerfreundliche Eingabe ---
class GroupedDescriptorEnum(str, Enum):
    laerm = "Lärm"
    verkehr = "Verkehr"
    strassenzustand = "Straßenzustand"
    infrastruktur = "Infrastruktur"
    gebaeudezustand = "Gebäudezustand"
    sauberkeit = "Sauberkeit"
    sonstiges = "Sonstiges"

GROUPED_DESCRIPTOR_MAPPING = {
    "Lärm": 'Loud Music/Party',            # Wählt einen repräsentativen Descriptor für die Gruppe
    "Verkehr": 'Blocked Driveway',
    "Straßenzustand": 'Street Condition',
    "Infrastruktur": 'Water System',
    "Gebäudezustand": 'HEAT/HOT WATER',
    "Sauberkeit": 'UNSANITARY CONDITION',
    "Sonstiges": 'Consumer Complaint'
}

def load_mapping_data():
    """Lädt die Mapping-Dateien beim Start."""
    global ZIP_BOROUGH_MAPPING, BOROUGH_DEMOGRAPHICS_MAPPING, DEMOGRAPHICS_FALLBACK
    
    # Pfade zu den Datendateien
    zip_path = Path(__file__).parent.parent / 'data' / 'nyc-zip-codes.csv'
    demographics_path = Path(__file__).parent.parent / 'data' / 'bevoelkerungsgruppen.csv'

    # Lade ZIP -> Borough Mapping
    if zip_path.exists():
        zip_df = pd.read_csv(zip_path)
        ZIP_BOROUGH_MAPPING = dict(zip(zip_df['ZipCode'].astype(str), zip_df['Borough'].str.upper()))
        print(f"✅ ZIP-to-Borough mapping loaded with {len(ZIP_BOROUGH_MAPPING)} entries.")
    else:
        print(f"⚠️ WARNING: ZIP code mapping file not found at {zip_path}. Borough inference will fail.")

    # Lade Borough -> Demographics Mapping
    if demographics_path.exists():
        demographics_df = pd.read_csv(demographics_path)
        demographics_df['Stadtteil'] = demographics_df['Stadtteil'].str.upper()
        # Berechne den Durchschnitt als Fallback
        fallback_data = demographics_df[['Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']].mean().to_dict()
        DEMOGRAPHICS_FALLBACK.update(fallback_data)

        BOROUGH_DEMOGRAPHICS_MAPPING = demographics_df.set_index('Stadtteil').to_dict('index')
        print(f"✅ Borough-to-Demographics mapping loaded for {len(BOROUGH_DEMOGRAPHICS_MAPPING)} boroughs.")
        print(f"    Fallback demographics set to: {DEMOGRAPHICS_FALLBACK}")
    else:
        print(f"⚠️ WARNING: Demographics file not found at {demographics_path}. Using zero-fallback.")

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global BOOSTER
    print("--- Application Startup ---")
    
    # Lade alle Mapping-Daten
    load_mapping_data()

    # Lade das ML-Modell
    try:
        print(f"Attempting to load model from: {SPECIFIC_MODEL_PATH}")
        if not SPECIFIC_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {SPECIFIC_MODEL_PATH}")
        BOOSTER = xgb.Booster()
        BOOSTER.load_model(SPECIFIC_MODEL_PATH)
        print("✅ XGBoost Booster loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. API will not be able to make predictions.")
        print(f"    Error: {e}")
        traceback.print_exc()
        BOOSTER = None
    yield
    print("--- Application Shutdown ---")

# --- App Initialization ---
app = FastAPI(
    title="NYC 311 Complaint Classifier API (Final Version)",
    description="Eine finale, benutzerfreundliche API zur Klassifizierung von NYC 311 Beschwerden. Stadtteil und Bevölkerungsdaten werden automatisch aus der Postleitzahl abgeleitet, und der Beschwerde-Descriptor kann aus einer vereinfachten Liste von Kategorien ausgewählt werden.",
    version="5.0.0",
    lifespan=lifespan
)

# --- API Data Models ---
class Complaint(BaseModel):
    # Benutzer muss nur diese Felder angeben
    Agency: str = "DSNY"
    Location_Type: str = "Street/Sidewalk"
    Incident_Zip: str
    Vehicle_Type: str = "N/A"
    # Descriptor ist jetzt eine gruppierte Kategorie
    Descriptor: GroupedDescriptorEnum = GroupedDescriptorEnum.strassenzustand
    
    # NEU: Nur noch ein Datumsfeld für die Eingabe
    created_date: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Borough ist jetzt optional
    Borough: Optional[str] = None

    class Config:
        json_schema_extra = { "example": { "Agency": "DSNY", "Location_Type": "Street/Sidewalk", "Incident_Zip": "10007", "Descriptor": "Straßenzustand", "created_date": "2024-07-01T10:30:00" } }
        populate_by_name = True

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return { "message": "NYC 311 Complaint Classifier API - Final Version", "model_loaded": "Yes" if BOOSTER else "No" }

@app.post("/predict", tags=["Prediction"])
def predict(complaint: Complaint):
    if not BOOSTER:
        raise HTTPException(status_code=503, detail="Model not loaded. Server is not ready.")

    try:
        input_dict = complaint.dict()
        
        # --- Automatische Datenanreicherung ---
        zip_code_str = str(input_dict.get("Incident_Zip", "")).strip()

        # 1. Borough ableiten, falls nicht vorhanden
        if not input_dict.get("Borough"):
            inferred_borough = ZIP_BOROUGH_MAPPING.get(zip_code_str)
            if not inferred_borough:
                print(f"WARNING: ZIP code {zip_code_str} not found in mapping. Using 'Unspecified'.")
                inferred_borough = 'Unspecified'
            input_dict["Borough"] = inferred_borough
            print(f"Inferred Borough: {inferred_borough}")

        # 2. Demografiedaten hinzufügen
        borough_for_demographics = input_dict["Borough"].upper()
        demographics = BOROUGH_DEMOGRAPHICS_MAPPING.get(borough_for_demographics, DEMOGRAPHICS_FALLBACK)
        input_dict.update(demographics)
        print(f"Added demographics for {borough_for_demographics}: {demographics}")
        # --- Ende der Anreicherung ---
        
        # --- NEU: Datums-Features und Duration aus 'created_date' generieren ---
        input_df = pd.DataFrame([input_dict])
        
        # Dauer berechnen (Zeit von 'created_date' bis jetzt)
        created_datetime = pd.to_datetime(input_df['created_date'].iloc[0])
        duration = datetime.now() - created_datetime
        input_df['duration_days'] = duration.total_seconds() / (60 * 60 * 24)
        print(f"Calculated duration: {input_df['duration_days'].iloc[0]:.2f} days")

        # Datums-Features extrahieren
        date_transformer = DateFeatureTransformer(date_column_name='created_date')
        input_df = date_transformer.transform(input_df)
        print(f"Extracted date features: hour, month, dayofweek")
        
        # Manuelles Encoding der textuellen Features
        encoded_dict = {}
        encoded_dict['Agency'] = AGENCY_MAPPING.get(input_df['Agency'].iloc[0], -1)
        encoded_dict['Location_Type'] = LOCATION_TYPE_MAPPING.get(input_df['Location_Type'].iloc[0], -1)
        encoded_dict['Borough'] = BOROUGH_MAPPING.get(input_df['Borough'].iloc[0], -1)
        encoded_dict['Vehicle_Type'] = VEHICLE_TYPE_MAPPING.get(input_df['Vehicle_Type'].iloc[0], -1)
        
        # Übersetze gruppierten Descriptor zum spezifischen Descriptor für das Modell
        grouped_descriptor = input_df['Descriptor'].iloc[0]
        specific_descriptor = GROUPED_DESCRIPTOR_MAPPING.get(grouped_descriptor, 'sonstiges')
        encoded_dict['Descriptor_encoded'] = DESCRIPTOR_MAPPING.get(specific_descriptor, DESCRIPTOR_MAPPING.get('sonstiges', 99))
        print(f"Grouped descriptor '{grouped_descriptor}' mapped to specific descriptor '{specific_descriptor}' -> encoded as {encoded_dict['Descriptor_encoded']}")
        
        # Numerische Features übernehmen
        encoded_dict['Incident_Zip'] = int(zip_code_str) if zip_code_str.isdigit() else 0
        
        # Übernehme die neu generierten Features
        for key in ['duration_days', 'Created_Date_hour', 'Created_Date_month', 'Created_Date_dayofweek', 'Weisse', 'Afroamerikaner', 'Asiaten', 'Hispanics']:
            encoded_dict[key] = input_df[key].iloc[0]
        
        # DataFrame erstellen und Spalten in der korrekten Reihenfolge anordnen
        final_input_data = pd.DataFrame([encoded_dict])
        final_input_data = final_input_data[FEATURE_NAMES]
        
        # DMatrix erstellen und Vorhersage
        dmatrix = xgb.DMatrix(final_input_data, feature_names=FEATURE_NAMES)
        prediction_proba = BOOSTER.predict(dmatrix)
        
        predicted_class_index = int(np.argmax(prediction_proba[0]))
        confidence_score = float(np.max(prediction_proba[0]))
        predicted_class_name_specific = CLASS_MAPPING.get(predicted_class_index, "Unknown Class")
        
        # --- NEU: Finale Gruppierung der Ausgabe ---
        predicted_class_group = OUTPUT_GROUPING_MAPPING.get(predicted_class_name_specific, "Nicht kategorisiert")

        return {
            "predicted_complaint_group": predicted_class_group,
            "predicted_complaint_type_specific": predicted_class_name_specific,
            "confidence_score": confidence_score,
            "inferred_borough": input_dict["Borough"],
            "inferred_demographics": demographics,
            "internal_descriptor_used": specific_descriptor,
            "calculated_duration_days": encoded_dict['duration_days']
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# To run this API:
# uvicorn src.main_api_final:app --reload 