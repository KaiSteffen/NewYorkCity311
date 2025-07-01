# Platzhalter für generate_test_data_final.py

import json
from datetime import datetime, timedelta

# Testfälle für die finale API.
# Die Erwartung ("expected") enthält jetzt die spezifische Vorhersage des Modells
# und die finale, für den Benutzer gruppierte Kategorie.
test_cases = [
    {
        "input": {
            "Agency": "DSNY", "Location_Type": "Street/Sidewalk", "Incident_Zip": "11221",
            "Descriptor": "Sauberkeit", "created_date": (datetime.now() - timedelta(days=2.1)).isoformat()
        },
        "expected": {"specific": "UNSANITARY CONDITION", "grouped": "Sauberkeit & Hygiene"}
    },
    {
        "input": {
            "Agency": "DOT", "Location_Type": "Street/Sidewalk", "Incident_Zip": "11238",
            "Descriptor": "Straßenzustand", "created_date": (datetime.now() - timedelta(days=14.5)).isoformat()
        },
        "expected": {"specific": "Street Light Condition", "grouped": "Zustand öffentlicher Raum"}
    },
    {
        "input": {
            "Agency": "NYPD", "Location_Type": "Street/Sidewalk", "Incident_Zip": "10467",
            "Descriptor": "Lärm", "created_date": (datetime.now() - timedelta(days=0.2)).isoformat()
        },
        "expected": {"specific": "Noise - Residential", "grouped": "Lärmbelästigung"}
    },
    {
        "input": {
            "Agency": "HPD", "Location_Type": "Residential Building/House", "Incident_Zip": "11432",
            "Descriptor": "Sauberkeit", "created_date": (datetime.now() - timedelta(days=8.0)).isoformat()
        },
        "expected": {"specific": "UNSANITARY CONDITION", "grouped": "Sauberkeit & Hygiene"}
    },
    {
        "input": {
            "Agency": "NYPD", "Location_Type": "Residential Building/House", "Incident_Zip": "10001",
            "Descriptor": "Lärm", "created_date": (datetime.now() - timedelta(days=0.5)).isoformat()
        },
        "expected": {"specific": "Noise - Residential", "grouped": "Lärmbelästigung"}
    },
    {
        "input": {
            "Agency": "DSNY", "Location_Type": "Residential Building/House", "Incident_Zip": "11375",
            "Descriptor": "Sauberkeit", "created_date": (datetime.now() - timedelta(days=3.0)).isoformat()
        },
        "expected": {"specific": "UNSANITARY CONDITION", "grouped": "Sauberkeit & Hygiene"}
    },
    {
        "input": {
            "Agency": "DEP", "Location_Type": "Commercial", "Incident_Zip": "10009",
            "Descriptor": "Infrastruktur", "created_date": (datetime.now() - timedelta(days=5.5)).isoformat()
        },
        "expected": {"specific": "Water System", "grouped": "Gebäude & Versorgung"}
    },
    {
        "input": {
            "Agency": "DOB", "Location_Type": "Residential Building/House", "Incident_Zip": "10451",
            "Descriptor": "Verkehr", "created_date": (datetime.now() - timedelta(days=1.2)).isoformat()
        },
        "expected": {"specific": "Blocked Driveway", "grouped": "Verkehrsprobleme"}
    },
    {
        "input": {
            "Agency": "DOT", "Location_Type": "Street/Sidewalk", "Incident_Zip": "10301",
            "Descriptor": "Straßenzustand", "created_date": (datetime.now() - timedelta(days=7.7)).isoformat()
        },
        "expected": {"specific": "Street Condition", "grouped": "Zustand öffentlicher Raum"}
    },
    {
        "input": {
            "Agency": "DSNY", "Location_Type": "Street/Sidewalk", "Incident_Zip": "10002",
            "Descriptor": "Verkehr", "created_date": (datetime.now() - timedelta(days=0.9)).isoformat()
        },
        "expected": {"specific": "Blocked Driveway", "grouped": "Verkehrsprobleme"}
    }
]

print("# Testdaten für die finale API (mit gruppierter Ausgabe)\n")
for i, case in enumerate(test_cases, 1):
    print(f"// --- Testfall {i} ---")
    print(f"// Eingabe:")
    print(json.dumps(case["input"], ensure_ascii=False, indent=2))
    print(f"// Erwartete Ausgabe:")
    print(f"//   -> Spezifischer Typ: {case['expected']['specific']}")
    print(f"//   -> Gruppierter Typ:  {case['expected']['grouped']}\n") 