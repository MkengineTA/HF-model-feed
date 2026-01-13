# Edge AI Scout & Specialist Model Monitor

**Edge AI Scout** is an automated, Python-based pipeline designed to discover, filter, and analyze specialized AI models on Hugging Face daily. It is specifically built to identify **Specialist Models** suitable for **Edge AI** and **Manufacturing** applications, filtering out general-purpose chat models, generative art, and irrelevant content.

## 🚀 Features

### 1. Multi-Source Discovery
The scout aggregates models from four strategic sources:
*   **Recently Created**: Scans brand new repositories (`sort=createdAt`).
*   **Recently Updated**: Finds models with fresh updates (`sort=lastModified`). If a model is already tracked but updated, it triggers a re-analysis (Delta Check).
*   **Trending Models**: Identifies models currently gaining traction in the community.
*   **Daily Papers**: Scans daily published research papers for linked model implementations.

### 2. Intelligent Filter Cascade (v2)
Before any expensive LLM analysis, models pass through a strict, multi-stage filter:

*   **Namespace Policy**:
    *   **Whitelist**: Trusted organizations (e.g., Google, Nvidia, Qwen) bypass some quality gates.
    *   **Blacklist**: Known quantization spammers or irrelevant users are skipped immediately.
*   **Hard Scope Filters**:
    *   **Generative Visuals**: Excludes Text-to-Image, Diffusion, 3D, and ComfyUI models.
    *   **Robotics/VLA**: Excludes pure robotics policies *unless* they are whitelisted as Vision/Inspection/VQA.
    *   **Exports/Conversions**: Excludes ONNX, GGUF, GPTQ, and MergeKit models using regex pattern matching.
    *   **NSFW**: Strict exclusion of adult content.
    *   **Parameters**: Defaults to <40B parameters (configurable).
*   **Quality Gate**:
    *   Rejects boilerplate READMEs ("More Information Needed").
    *   Calculates an `Info Score` based on YAML metadata, tags, and text density. Low-quality user repositories are skipped.

### 3. LLM Agent Workflow & Evidence Validation
The core analysis engine uses an LLM (Local or Cloud) acting as a "Strict Analyst".

1.  **Context Extraction**: Parses README text and YAML frontmatter.
2.  **Analysis**:
    *   **Categorization**: Identifies model type (Base, Adapter, Finetune).
    *   **Delta Analysis**: For finetunes, extracts *exactly* what changed (Dataset, Objective) and the value proposition vs. the base model.
    *   **Scoring**: Scores suitability for Manufacturing/Edge (1-10).
3.  **Evidence Gate**: The LLM must provide **direct quotes** from the README to support its claims. The system programmatically validates these quotes. If a quote is missing or hallucinated, the model is flagged or downgraded.

### 4. Robust Reporting
*   **Run Statistics**: Tracks total candidates, skipped models (with reasons), and analyzed count.
*   **Markdown Newsletter**: Generates a structured report with "Cards" for each model, sorted by Specialist Score.
*   **Email Dispatch**: Converts the Markdown report to Outlook-optimized HTML and sends it via SMTP (Gmail).
*   **Persistence**: Stores model status, metadata, and skip traces in SQLite for historical tracking and debugging.

---

## 🛠 Installation & Configuration

### Prerequisites
*   Python 3.9+
*   Hugging Face Account (for API Token)

### Setup

1.  **Clone Repository**:
    ```bash
    git clone <repo-url>
    cd edge-ai-scout
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration (.env)**:
    Copy the example and customize it:
    ```bash
    cp .env.example .env
    ```

### Configuration Variables (`.env`)

| Variable | Description |
| :--- | :--- |
| `HF_TOKEN` | **Required**. Hugging Face User Access Token (Read). |
| `DB_PATH` | Path to SQLite database (default: `models.db`). |
| `LLM_API_URL` | Endpoint for LLM (e.g., OpenRouter or local Ollama). |
| `LLM_MODEL` | Model ID (e.g., `qwen/qwen-2.5-72b-instruct`). |
| `LLM_API_KEY` | API Key for the LLM provider. |
| `LLM_ENABLE_REASONING` | Set to `True` if using reasoning models (e.g., R1, o1). |
| `LLM_REASONING_EFFORT` | Reasoning effort level for applicable models: `low`, `medium`, or `high` (default: `medium`). |
| `SMTP_USER` | Gmail address for sending reports. |
| `SMTP_PASS` | Gmail App Password (not your login password). |
| `RECEIVER_MAIL` | Destination email address. |
| `MODEL_NAME_DUPLICATE_BLOCK_LIMIT` | Block further processing of a model name when its occurrence count in a single run reaches this limit (default: `3`; setting it to `3` allows only 2 occurrences to be processed—the 3rd is blocked). |

---

## 🚀 Usage

### Manual Run (Testing)
To test the pipeline without saving to DB ("Dry Run") and force an email dispatch:

```bash
python main.py --limit 5 --dry-run --force-email
```

*   `--limit <n>`: Fetches max `n` models per source (Safety ceiling).
*   `--dry-run`: Performs analysis but does not commit changes to the database.
*   `--force-email`: Sends the email report even in Dry Run mode (usually suppressed).

### Production (Cronjob)
Run the scout daily (e.g., at 06:00 AM). It automatically handles incremental updates (fetching only models created/updated since the last successful run).

```bash
0 6 * * * cd /path/to/edge-ai-scout && /usr/bin/python3 main.py >> scout.log 2>&1
```

---

## 🏗 Architecture Details

### Trust Tiers & Author Detection
The system caches author metadata (User vs. Organization) to reduce API calls and apply differential filtering:
*   **Tier 3 (Trusted Org)**: Organizations detected via `/avatar` endpoint. Quality gates are relaxed (we review "thin" READMEs from trusted orgs).
*   **Tier 2 (Strong User)**: Users with >200 followers or PRO status.
*   **Tier 1 (Normal User)**: Strict quality gates apply.

### Caching Strategy
*   **Author Cache**: Stores namespace types (Org/User) for 14 days. 'Unknown' status (404) is NOT cached to allow immediate re-validation if an account is created.
*   **Incremental Fetching**: The `metadata` table stores the `last_run` timestamp. Subsequent runs fetch only models newer than this timestamp.

---

## 📊 Output Format

The generated report highlights key technical details:

### [Model Name]
*   **Type**: Finetune | **Score**: 9/10 | **Author**: Organization (Tier 3)
*   **Blurb**: concise technical summary (approx. 100 words).
*   **Key Facts**: Bullet points on architecture and training.
*   **Delta**: Specific changes from the base model and why they matter.
*   **Edge Readiness**: VRAM requirements and quantization status.
*   **Manufacturing Fit**: Specific use cases (e.g., defect detection).
*   **Confidence**: High/Medium/Low (based on evidence validation).

---

## 🧪 Development

Run the test suite to verify filtering logic and integrations:

```bash
python -m unittest discover .
```

---

## 📧 Multi-Recipient Newsletter Configuration

Edge AI Scout supports **multiple newsletter recipients** with per-recipient customization:

*   **Per-recipient send days**: Choose which days each recipient receives emails (e.g., Mon/Wed/Fri).
*   **Per-recipient coverage window**: Configure different time windows per day (e.g., 72h on Monday, 24h on Friday).
*   **Recipient type**: `debug` (full report with skip/warn stats) or `normal` (clean report without debug sections).
*   **Per-recipient language**: English (`en`) or German (`de`).

### Configuration

Set `NEWSLETTER_SUBSCRIBERS_JSON` in your `.env` file:

```env
NEWSLETTER_TIMEZONE=Europe/Berlin

NEWSLETTER_SUBSCRIBERS_JSON=[
  {
    "email": "alice@example.com",
    "type": "normal",
    "language": "en",
    "send_days": ["mon","wed","fri"],
    "default_window_hours": 24,
    "window_hours_by_day": {"mon": 72, "wed": 48}
  },
  {
    "email": "debug@example.com",
    "type": "debug",
    "language": "de",
    "send_days": ["mon","tue","wed","thu","fri"],
    "default_window_hours": 24
  }
]
```

### Backward Compatibility

If `NEWSLETTER_SUBSCRIBERS_JSON` is not set, the system falls back to using `RECEIVER_MAIL` as a single debug recipient with German language.

### Coverage Window (`processed_at`)

Digest coverage is based on the `processed_at` timestamp (when the pipeline processed/saved the model), not HF's `created_at` or `lastModified`. This enables generating longer-window digests (48h/72h) purely from SQLite without additional HF/LLM API calls.

---

## 🌐 Internationalization (i18n)

Edge AI Scout currently supports **English** (`en`) and **German** (`de`) for newsletter reports.

### Adding New Languages

To add support for a new language (e.g., French `fr`):

1.  **Update Subscriber Config Validation** (`config.py`):
    ```python
    SUPPORTED_LANGUAGES = {"en", "de", "fr"}
    ```

2.  **Extend Localized String Table** (`reporter.py`):
    Add French translations to the `LOCALIZED_STRINGS` dictionary:
    ```python
    "summary": {
        "de": "Zusammenfassung",
        "en": "Summary",
        "fr": "Résumé",
    },
    # ... add French for all keys
    ```

3.  **Extend LLM Output Schema** (`llm_client.py`):
    Update the LLM prompt to request French output alongside German and English:
    ```
    "newsletter_blurb": {"de": "...", "en": "...", "fr": "..."},
    ```
    Note: This increases token usage but keeps request count unchanged.

4.  **Extend Email Subject Localization** (`mailer.py`):
    ```python
    EMAIL_SUBJECTS = {
        "de": "Edge AI Scout Report - {date}",
        "en": "Edge AI Scout Report - {date}",
        "fr": "Rapport Edge AI Scout - {date}",
    }
    ```

5.  **Update Tests**:
    Add tests to verify language selection and report rendering for the new language.
