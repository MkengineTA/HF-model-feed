# Edge AI Scout & Specialist Model Monitor

**Edge AI Scout** ist ein automatisiertes, Python-basiertes Tool zur täglichen Entdeckung, Filterung und Analyse neuer KI-Modelle auf Hugging Face. Es ist speziell darauf ausgelegt, **Specialist Models** für **Edge AI** und **Manufacturing** (Fertigung) zu identifizieren.

Das System scannt mehrere Quellen auf Hugging Face, filtert ungeeignete Modelle (zu groß, unsicher, irrelevant) und nutzt ein LLM (Lokal oder Cloud), um das Potential für industrielle Anwendungen zu bewerten.

## 🚀 Features

### 1. Multi-Source Discovery
Das Tool verlässt sich nicht nur auf eine Quelle, sondern aggregiert Modelle aus drei Bereichen:
*   **Neueste Uploads**: Scannt die `recently created` Modelle.
*   **Trending Models**: Identifiziert Modelle, die aktuell in der Community an Fahrt gewinnen.
*   **Daily Papers**: Durchsucht täglich veröffentlichte Forschungspapiere nach verknüpften Modell-Implementierungen (oft High-Quality Research Code).

### 2. Intelligente Filter-Kaskade
Bevor ein Modell teuer analysiert wird, durchläuft es strenge Filter:
*   **Parameter-Limit (< 10B)**:
    *   Prüfung via `safetensors` Metadaten.
    *   Fallback auf Regex im Modellnamen (z.B. "7b", "1.5B").
    *   **NEU**: Heuristischer Fallback über Dateigrößen (Proxy: >20GB ≈ >10B Params).
*   **Sicherheits-Check**:
    *   Prüfung des Hugging Face `securityFileStatus`.
    *   Modelle mit "unsafe" Scans (Malware, Pickle-Exploits) werden sofort verworfen.
*   **Format & Inhalt**:
    *   Ausschluss reiner Quantisierungen (GGUF, AWQ, EXL2), um Original-Modelle zu finden.
    *   Filterung unerwünschter Inhalte (Roleplay, NSFW, Uncensored) via Tags und Namens-Matching.

### 3. LLM-basierte Analyse
*   Kompatibel mit **lokalen LLMs** (Ollama, LM Studio) und **Cloud-Providern** (OpenRouter, OpenAI).
*   Analysiert READMEs auf:
    *   **Specialist Score (1-10)**: Wie stark ist das Modell spezialisiert?
    *   **Manufacturing Potential**: Erkennt Use-Cases wie Defekterkennung, Predictive Maintenance, PDF-Extraktion.
    *   **Kategorisierung**: Vision, Code, Reasoning, etc.

### 4. Reporting
*   **Markdown Newsletter**: Täglicher Bericht (`report_YYYY-MM-DD.md`) mit "High Potential" Highlights.
*   **CSV Export**: Strukturierte Liste (`labeling_pending.csv`) für manuelles Review oder Active Learning.
*   **SQLite Datenbank**: Persistente Speicherung zur Deduplizierung (verhindert doppelte Analysen).

---

## 🛠 Installation

### Voraussetzungen
*   Python 3.8+
*   Ein Hugging Face Account (für den API Token)

### Setup

1.  **Repository klonen**:
    ```bash
    git clone <repo-url>
    cd <repo-folder>
    ```

2.  **Abhängigkeiten installieren**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Konfiguration (.env)**:
    Kopieren Sie die Vorlage und passen Sie sie an:
    ```bash
    cp .env.example .env
    ```

---

## ⚙️ Konfiguration

Die Steuerung erfolgt über Umgebungsvariablen in der `.env` Datei.

### Basis-Konfiguration
| Variable | Beschreibung |
| :--- | :--- |
| `HF_TOKEN` | **Pflichtfeld**. Ihr Hugging Face User Access Token (Read permissions). |
| `DB_PATH` | Pfad zur SQLite Datenbank (Standard: `models.db`). |

### LLM Backend Auswahl

Sie können wählen, ob Sie ein lokales Modell (kostenlos, privat) oder eine Cloud-API (mächtiger, kostenpflichtig) nutzen.

#### Option A: Lokales LLM (z.B. Ollama)
Ideal für Datenschutz und kostenlosen Betrieb.
```ini
LLM_API_URL=http://localhost:11434/v1/chat/completions
LLM_MODEL=llama3
# LLM_API_KEY leer lassen
```

#### Option B: OpenRouter / OpenAI (Cloud)
Ideal für höhere Analyse-Qualität (z.B. mit GPT-4o oder großen Open-Source Modellen wie Llama-3-70B/Qwen-110B).
```ini
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=openai/gpt-oss-120b:free  # Oder jedes andere OpenRouter Modell
LLM_API_KEY=sk-or-your-key-here     # Ihr OpenRouter API Key

# Optional: Für OpenRouter Rankings / Statistiken
OR_SITE_URL=https://github.com/IhrUser/EdgeAIScout
OR_APP_NAME=Edge AI Scout
```

---

## 🚀 Nutzung

Das Tool ist als CLI-Anwendung konzipiert und kann manuell oder via Cronjob gestartet werden.

### Manueller Start
```bash
python main.py
```

### Parameter
*   `--limit <n>`: Anzahl der Modelle, die aus der "Recent"-Liste geladen werden (Standard: 200).
    *   *Hinweis: Trending und Papers werden unabhängig davon zusätzlich geladen.*
*   `--dry-run`: Führt den kompletten Scan und die Analyse durch, speichert aber **nichts** in die Datenbank. Gut zum Testen neuer Filter.

Beispiel:
```bash
python main.py --limit 50 --dry-run
```

### Automatisierung (Cronjob)
Um den Scout jeden Morgen um 06:00 Uhr laufen zu lassen:

```bash
0 6 * * * cd /pfad/zu/edge-ai-scout && /pfad/zu/python main.py >> scout.log 2>&1
```

---

## 📊 Output Formate

### 1. Markdown Report (`report_YYYY-MM-DD.md`)
Ein lesbarer Bericht, unterteilt in:
*   **High Potential**: Modelle mit Specialist Score > 7.
*   **Review Required**: Modelle mit externen Links aber zu kurzer Beschreibung auf HF.

### 2. CSV Export (`labeling_pending.csv`)
Dient als Input für Labeling-Tools oder Excel.
Spalten: `Modell-ID`, `Kategorie`, `Zusammenfassung`, `User-Label`.

### 3. Datenbank (`models.db`)
SQLite Datei. Speichert den Status (`processed`, `error`, `review_required`) und die JSON-Analyse jedes Modells.

---

## 🧪 Entwicklung & Tests

Das Projekt enthält eine umfassende Test-Suite (Unit & Integration).

```bash
# Alle Tests ausführen
python -m unittest discover .

# Nur Integrationstests (mocked API Flow)
python -m unittest test_integration.py
```
