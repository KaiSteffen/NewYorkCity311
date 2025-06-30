import subprocess
import sys
import json
import pytest
from pathlib import Path
import locale

# Finde das Hauptverzeichnis des Projekts, damit wir das Skript sicher finden können.
# Path(__file__) ist der Pfad zur aktuellen Testdatei.
# .parent.parent geht zwei Ebenen nach oben (von /tests/ nach /).
project_root = Path(__file__).parent.parent
script_path = project_root / 'generate_test_data_final.py'

def test_script_runs_without_error():
    """Stellt sicher, dass das Skript ohne Fehler durchläuft."""
    # Stelle sicher, dass das Skript existiert, bevor wir es ausführen
    if not script_path.exists():
        pytest.fail(f"Das zu testende Skript wurde nicht gefunden unter: {script_path}")

    try:
        # Führe das Skript als externen Prozess aus.
        # Wir übergeben den absoluten Pfad und setzen das Arbeitsverzeichnis.
        result = subprocess.run(
            [sys.executable, str(script_path)],  # Wichtig: Path-Objekt zu String konvertieren
            capture_output=True,
            text=True,
            check=True,
            encoding=locale.getpreferredencoding(False),
            cwd=project_root  # Führe das Skript vom Projekt-Stammverzeichnis aus
        )
        # Überprüft, ob der Exit-Code 0 ist (check=True)
        assert result.returncode == 0
        # Stellt sicher, dass eine Ausgabe vorhanden ist
        assert result.stdout
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Das Skript ist mit einem Fehler fehlgeschlagen:\n{e.stderr}")

def test_output_is_valid_json_and_structured_correctly():
    """
    Fängt die Ausgabe des Skripts ab und überprüft, ob jeder Testfall
    ein valides JSON-Objekt für die Eingabe ist und die erwartete Struktur hat.
    """
    if not script_path.exists():
        pytest.fail(f"Das zu testende Skript wurde nicht gefunden unter: {script_path}")
        
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=True,
        encoding=locale.getpreferredencoding(False),
        cwd=project_root
    )
    
    # Trenne die Ausgabe in einzelne Blöcke (jeder Block ist ein Testfall)
    # KORREKTUR: Wir splitten beim Separator und verwerfen den Header (das erste Element),
    # indem wir bei Index 1 mit der Verarbeitung beginnen.
    all_parts = result.stdout.split('// ---')
    output_blocks = [block.strip() for block in all_parts[1:] if block.strip()]
    
    assert len(output_blocks) > 0, "Keine Testfallblöcke in der Ausgabe gefunden."
    
    for i, block in enumerate(output_blocks, 1):
        # Zerlege den Block in seine Teile
        lines = block.split('\n')
        
        # Finde die Zeile, die mit '{' beginnt - das ist unser JSON
        json_str = ""
        json_started = False
        for line in lines:
            if line.strip().startswith('{'):
                json_started = True
            if json_started:
                json_str += line
                if line.strip().endswith('}'):
                    break # Ende des JSON-Objekts

        assert json_str, f"Kein JSON-Objekt im Block {i} gefunden."

        # Versuche, das JSON zu parsen
        try:
            data = json.loads(json_str)
            
            # Überprüfe die erwartete Struktur des JSON-Inputs
            assert "Agency" in data, f"Schlüssel 'Agency' fehlt im JSON von Block {i}"
            assert "Incident_Zip" in data, f"Schlüssel 'Incident_Zip' fehlt im JSON von Block {i}"
            assert "Descriptor" in data, f"Schlüssel 'Descriptor' fehlt im JSON von Block {i}"
            assert "created_date" in data, f"Schlüssel 'created_date' fehlt im JSON von Block {i}"
            
        except json.JSONDecodeError as e:
            pytest.fail(f"Fehler beim Parsen von JSON in Block {i}: {e}\nJSON-String war:\n{json_str}")

        # Überprüfe, ob die erwarteten Ausgabekommentare vorhanden sind
        assert any("Erwartete Ausgabe:" in line for line in lines), f"Erwartete Ausgabe-Kommentar fehlt in Block {i}"
        assert any("-> Spezifischer Typ:" in line for line in lines), f"Spezifischer Typ-Kommentar fehlt in Block {i}"
        assert any("-> Gruppierter Typ:" in line for line in lines), f"Gruppierter Typ-Kommentar fehlt in Block {i}"

