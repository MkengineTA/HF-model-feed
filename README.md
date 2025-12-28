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
1.  **Input**: Das komplette README (bis 32k Kontext) + HF Tags.
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

## 🛠 Installation & Konfiguration

Siehe `INSTALL.md` (oder vorherige README Sektionen, hier gekürzt).

### Konfiguration (.env)
```ini
HF_TOKEN=...
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=openai/gpt-oss-120b:free
LLM_API_KEY=...
```

---

## 🚀 Nutzung

```bash
python main.py --limit 100
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
