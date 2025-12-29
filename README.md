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
*   **Markdown Newsletter**: Listet alle verarbeiteten Modelle, sortiert nach Specialist Score.
*   **E-Mail Versand**: Generiert einen professionellen HTML-Report (Outlook-kompatibel) und versendet ihn via SMTP.
*   **CSV Export**: Strukturierte Liste für Labeling.
*   **SQLite Datenbank**: Speichert Status, Zeitstempel (Created/Modified) und Analyse-Ergebnisse.

---

## 🧠 LLM Anforderungen & Auswahl

Damit der Scout zuverlässig funktioniert, muss das gewählte LLM bestimmte technische Anforderungen erfüllen.

### 1. Context Window (Kontext-Fenster)
*   **Empfehlung:** **Mindestens 16k Tokens** (besser 32k).
*   **Grund:** Der Scout sendet bis zu **32.000 Zeichen** (ca. 8k - 10k Tokens) reinen README-Text.

### 2. Fähigkeiten (Capabilities)
*   **JSON Output (Kritisch):** Das Modell **muss** zuverlässig valides JSON generieren.
*   **Reasoning (Wichtig):** Für die **Delta-Analyse** muss das Modell verstehen, *warum* ein Finetune existiert.

### 3. Modell-Empfehlungen
*   **Cloud (OpenRouter):** `qwen/qwen-2.5-72b-instruct`, `meta-llama/llama-3.1-70b-instruct`.
*   **Lokal (Ollama):** `llama3.1:8b-fp16`, `mistral-nemo:12b`.

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
DB_PATH=models.db

# LLM Konfiguration
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=qwen/qwen-2.5-72b-instruct
LLM_API_KEY=sk-or-your-key-here
LLM_ENABLE_REASONING=True

# E-Mail Konfiguration (Beispiel Gmail)
SMTP_USER=ihre.email@gmail.com
SMTP_PASS=ihr-app-passwort
RECEIVER_MAIL=ziel.email@firma.com
```

---

## 📧 E-Mail Setup (Gmail)

Da Gmail den einfachen Login via Passwort deaktiviert hat, benötigen Sie ein **App-Passwort**:

1.  Loggen Sie sich in Ihr Google Konto ein.
2.  Gehen Sie zu **Sicherheit** -> **Bestätigung in zwei Schritten** (muss aktiviert sein).
3.  Scrollen Sie unten zu **App-Passwörter**.
4.  Erstellen Sie ein neues Passwort (Name: "Edge AI Scout").
5.  Kopieren Sie das 16-stellige Passwort (ohne Leerzeichen) in die `.env` unter `SMTP_PASS`.

Der generierte HTML-Report ist für **Outlook**, **Thunderbird** und **Gmail Web** optimiert (nutzt Inline-CSS und Tabellen-Layouts).

---

## ❓ FAQ & Logik

### Wie funktioniert der erste Run?
Wenn die Datenbank leer ist (erster Start), setzt der Scout den "letzten Run" automatisch auf **24 Stunden in der Vergangenheit**.
Er lädt also **nicht** alle Modelle seit Beginn der Zeit, sondern nur die des letzten Tages.

### Wie wird das Update-Intervall gesteuert?
Das Skript speichert am Ende jedes erfolgreichen Laufs einen Zeitstempel (`metadata` Tabelle). Beim nächsten Start werden nur Modelle geladen, die **nach** diesem Zeitstempel erstellt oder aktualisiert wurden.

---

## 🧪 Testing

Um das Setup zu testen, ohne die Datenbank zu verändern ("Dry Run"), aber trotzdem eine Test-E-Mail zu erhalten:

```bash
python main.py --limit 5 --dry-run --force-email
```
*   `--limit 5`: Lädt nur 5 Modelle (spart API-Calls/Kosten).
*   `--dry-run`: Speichert nichts in die DB (Modelle werden beim nächsten echten Run erneut verarbeitet).
*   `--force-email`: Erzwingt den E-Mail-Versand, der normalerweise im Dry-Run deaktiviert ist.

---

## 🚀 Nutzung (Produktion)

Starten Sie den Scan ohne Test-Flags:

```bash
python main.py
```

### Automatisierung (Cronjob)
```bash
0 6 * * * cd /pfad/zu/edge-ai-scout && /pfad/zu/python main.py >> scout.log 2>&1
```
