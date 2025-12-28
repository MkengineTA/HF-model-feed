# Edge AI Scout & Specialist Model Monitor

**Edge AI Scout** ist ein automatisiertes, Python-basiertes Tool zur täglichen Entdeckung, Filterung und Analyse neuer KI-Modelle auf Hugging Face. Es ist speziell darauf ausgelegt, **Specialist Models** für **Edge AI** und **Manufacturing** (Fertigung) zu identifizieren.

Das System scannt mehrere Quellen auf Hugging Face, filtert ungeeignete Modelle (zu groß, unsicher, irrelevant) und nutzt ein LLM (Lokal oder Cloud), um das Potential für industrielle Anwendungen zu bewerten.

## 🚀 Features

### 1. Multi-Source Discovery (4 Säulen)
Das Tool aggregiert Modelle aus vier strategischen Quellen:
*   **Recently Created**: Scannt brandneue Repositories (`sort=createdAt`).
*   **Recently Updated**: Findet Modelle mit frischen Updates (`sort=lastModified`). Wenn ein Modell bereits bekannt ist, aber geupdated wurde, wird es neu analysiert (Delta-Check).
*   **Trending Models**: Identifiziert Modelle, die aktuell in der Community populär sind.
*   **Daily Papers**: Durchsucht täglich veröffentlichte Forschungspapiere nach verknüpften Modell-Implementierungen.

### 2. Intelligente Filter-Kaskade
Bevor ein Modell teuer analysiert wird, durchläuft es strenge Filter:
*   **Parameter-Limit (< 10B)**: Metadaten, Regex oder Dateigrößen-Heuristik.
*   **Sicherheits-Check**: Ausschluss von Modellen mit "unsafe" Scans.
*   **Format & Inhalt**: Ausschluss von Quantisierungen und unerwünschten Inhalten.

### 3. LLM Agent Workflow
Der Kern des Scouts ist ein intelligenter LLM-Agent.
**Workflow:**
1.  **Input**: Das komplette README (bis 32k Zeichen) + HF Tags.
2.  **Prompting**: Der Agent erhält eine spezifische Persona ("Expert AI Researcher") und Instruktionen zur technischen Tiefe.
3.  **Analyse-Schritte**:
    *   **Identifikation**: Ist es ein Base Model, Adapter (LoRA) oder Finetune?
    *   **Delta-Analyse**: Bei Adaptern wird explizit herausgearbeitet, was sich zum Basismodell geändert hat (Dataset, Zielaufgabe) und was der Mehrwert ist.
    *   **Scoring**: Bewertung (1-10) der Eignung für Manufacturing/Edge.
4.  **Output**: Strukturiertes JSON für die Weiterverarbeitung.

### 4. Reporting
*   **Markdown Newsletter**: Listet alle verarbeiteten Modelle, sortiert nach Specialist Score. Enthält detaillierte technische Zusammenfassungen und Delta-Analysen.
*   **CSV Export**: Strukturierte Liste für Labeling.
*   **SQLite Datenbank**: Speichert Status, Zeitstempel (Created/Modified) und Analyse-Ergebnisse.

---

## 🧠 LLM Anforderungen & Auswahl

Damit der Scout zuverlässig funktioniert, muss das gewählte LLM bestimmte technische Anforderungen erfüllen.

### 1. Context Window (Kontext-Fenster)
*   **Empfehlung:** **Mindestens 16k Tokens** (besser 32k).
*   **Grund:** Der Scout sendet bis zu **32.000 Zeichen** (ca. 8k - 10k Tokens) reinen README-Text plus System-Prompts. Ein Modell mit nur 4k oder 8k Kontext würde hier abschneiden oder halluzinieren.
*   **Einstellung bei `llama-server`**: Nutzen Sie den Parameter `-c 16384` oder höher.

### 2. Fähigkeiten (Capabilities)
*   **JSON Output (Kritisch):** Das Modell **muss** zuverlässig valides JSON generieren können. Der Scout nutzt Prompt-Engineering, um JSON zu erzwingen. Modelle, die dazu neigen, "Hier ist dein JSON:" davor zu schreiben (Chat-Fluff), sind weniger geeignet (obwohl der Code versucht, dies zu parsen).
*   **Reasoning (Wichtig):** Für die **Delta-Analyse** muss das Modell verstehen, *warum* ein Finetune existiert. Es muss technische Details aus langen Texten extrahieren und abstrahieren.
*   **Tool Calling:** Wird **nicht** benötigt. Der Scout nutzt Standard-Completion mit JSON-Schema im Prompt.

### 3. Modell-Empfehlungen
*   **Cloud (OpenRouter):**
    *   `qwen/qwen-2.5-72b-instruct` (Hervorragend für JSON & Coding Tasks, günstig).
    *   `meta-llama/llama-3.1-70b-instruct` (Sehr starkes Reasoning, 128k Kontext).
    *   `openai/gpt-4o-mini` (Schnell, günstig, sehr zuverlässiges JSON).
*   **Lokal (Ollama / Llama.cpp):**
    *   `llama3.1:8b-fp16` (Gutes Minimum, Kontext auf 16k setzen!).
    *   `mistral-nemo:12b` (Sehr gut für technische Texte, großes Kontextfenster).
    *   `qwen2.5:14b` (Aktueller Preis/Leistungs-Sieger für Structured Output).

---

## 🛠 Installation & Konfiguration

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

### Konfiguration (.env)
```ini
HF_TOKEN=...
# Datenbank Pfad
DB_PATH=models.db

# LLM Konfiguration (Beispiel OpenRouter)
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=qwen/qwen-2.5-72b-instruct
LLM_API_KEY=sk-or-your-key-here
# Optional: App Name für OpenRouter Statistiken
OR_APP_NAME=Edge AI Scout
OR_SITE_URL=https://github.com/IhrUser/EdgeAIScout
```

---

## 🚀 Nutzung

```bash
python main.py --limit 100
```

### Parameter
*   `--limit <n>`: Anzahl der Modelle pro Quelle (Standard: 100).
*   `--dry-run`: Test-Modus (keine DB-Speicherung).

### Automatisierung (Cronjob)
```bash
0 6 * * * cd /pfad/zu/edge-ai-scout && /pfad/zu/python main.py >> scout.log 2>&1
```

---

## 📊 Output Beispiel (Report)

### [Manufacturing-BERT-v2](https://huggingface.co/...)
- **Score:** 9/10
- **Typ:** Finetune
- **Basis:** bert-base-uncased
- **Zusammenfassung:** Ein auf 50.000 Wartungsprotokollen nachtrainiertes BERT Modell...
- **Das Delta:** Im Gegensatz zum Basismodell versteht dieses Modell spezifische Fehlercodes (ISO-1234) und Maschinenteil-Bezeichnungen.
- **Tags:** #manufacturing #nlp
- **Daten-Quelle:** README / Metadaten-Inferenz

---
