# Edge AI Scout & Specialist Model Monitor

Ein automatisiertes Tool zum Scannen, Filtern und Analysieren neuer Hugging Face Modelle, spezialisiert auf Edge AI und Manufacturing.

## Features

*   **Automatischer Scan**: Überprüft täglich die neuesten Modelle auf Hugging Face.
*   **Intelligente Filterung**:
    *   Ausschluss von Modellen > 10B Parametern.
    *   Filterung von Quantisierungen und unerwünschten Inhalten (NSFW, Roleplay).
    *   Deduplizierung bereits gescannter Modelle.
*   **LLM Analyse**: Nutzt ein lokales LLM (via OpenAI API / Ollama), um Modelle auf Spezialisierung und Manufacturing-Potential zu prüfen.
*   **Reporting**: Generiert tägliche Markdown-Berichte und CSV-Exporte für manuelles Labeling.

## Installation

1.  **Repository klonen**:
    ```bash
    git clone <repo-url>
    cd <repo-folder>
    ```

2.  **Abhängigkeiten installieren**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Konfiguration**:
    Erstellen Sie eine `.env` Datei basierend auf `.env.example`:
    ```bash
    cp .env.example .env
    ```
    Passen Sie die Werte an:
    *   `HF_TOKEN`: Ihr Hugging Face API Token.
    *   `LLM_API_URL`: URL zu Ihrem lokalen LLM (z.B. Ollama oder vLLM).
    *   `LLM_MODEL`: Name des zu verwendenden Modells.

## Nutzung

Starten Sie den Scan-Vorgang:

```bash
python main.py
```

Optionale Parameter:
*   `--limit`: Anzahl der zu scannenden Modelle (Standard: 200).
*   `--dry-run`: Führt den Prozess aus, ohne in die Datenbank zu schreiben.

## Output

*   **Reports**: Zu finden im Root-Verzeichnis als `report_YYYY-MM-DD.md`.
*   **CSV**: `labeling_pending.csv` enthält neue Modelle für manuelles Review.
*   **Datenbank**: `models.db` speichert den Status aller gescannten Modelle.

## Tests

Ausführen der Tests:

```bash
python -m unittest test_suite.py
python -m unittest test_integration.py
```
