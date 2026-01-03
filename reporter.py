# reporter.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List, Literal
from collections import Counter
import csv
import os
import math

import config
from run_stats import RunStats

# Localization string table for report headings
LOCALIZED_STRINGS: Dict[str, Dict[str, str]] = {
    "report_title": {
        "de": "EdgeAIScout Report",
        "en": "EdgeAIScout Report",
    },
    "summary": {
        "de": "Zusammenfassung",
        "en": "Summary",
    },
    "discovered_unique": {
        "de": "Entdeckt (eindeutig)",
        "en": "Discovered (unique)",
    },
    "queued_new_updated": {
        "de": "In Warteschlange (neu/aktualisiert)",
        "en": "Queued (new/updated)",
    },
    "noop_unchanged": {
        "de": "Unverändert (bereits erfasst)",
        "en": "No-op (unchanged/already tracked)",
    },
    "included_in_report": {
        "de": "Im Report enthalten",
        "en": "Included in report",
    },
    "skipped": {
        "de": "Übersprungen",
        "en": "Skipped",
    },
    "warnings": {
        "de": "Warnungen",
        "en": "Warnings",
    },
    "llm_attempted": {
        "de": "LLM versucht",
        "en": "LLM attempted",
    },
    "llm_succeeded": {
        "de": "LLM erfolgreich",
        "en": "LLM succeeded",
    },
    "llm_failed": {
        "de": "LLM fehlgeschlagen",
        "en": "LLM failed",
    },
    "top_skip_reasons": {
        "de": "Top Skip-Gründe",
        "en": "Top skip reasons",
    },
    "top_warning_reasons": {
        "de": "Top Warnungsgründe",
        "en": "Top warning reasons",
    },
    "top_skipped_uploaders": {
        "de": "Top übersprungene Uploader",
        "en": "Top skipped uploaders",
    },
    "processed_models_overview": {
        "de": "Verarbeitete Modelle (Übersicht)",
        "en": "Processed models (overview)",
    },
    "top_included_uploaders": {
        "de": "Top enthaltene Uploader",
        "en": "Top included uploaders",
    },
    "models": {
        "de": "Modelle",
        "en": "Models",
    },
    "no_models_criteria": {
        "de": "Keine Modelle erfüllen die Kriterien für diesen Report.",
        "en": "No models met the criteria for this report.",
    },
    "none": {
        "de": "(keine)",
        "en": "(none)",
    },
    "detailed_analysis": {
        "de": "Detaillierte Analyse",
        "en": "Detailed analysis",
    },
    "no_processed_models": {
        "de": "Keine verarbeiteten Modelle anzuzeigen.",
        "en": "No processed models to display.",
    },
    "key_facts": {
        "de": "Schlüsselfakten",
        "en": "Key facts",
    },
    "delta": {
        "de": "Delta",
        "en": "Delta",
    },
    "what_changed": {
        "de": "Was ist neu?",
        "en": "What's new?",
    },
    "why_relevant": {
        "de": "Warum relevant?",
        "en": "Why it matters?",
    },
    "manufacturing": {
        "de": "Fertigung",
        "en": "Manufacturing",
    },
    "use_cases": {
        "de": "Anwendungsfälle",
        "en": "Use cases",
    },
    "evidence_quotes": {
        "de": "Belege (Zitate)",
        "en": "Evidence (quotes)",
    },
    "claim": {
        "de": "Behauptung",
        "en": "Claim",
    },
    "quote": {
        "de": "Zitat",
        "en": "Quote",
    },
    "uploader": {
        "de": "Uploader",
        "en": "uploader",
    },
    "score": {
        "de": "Score",
        "en": "score",
    },
    "type": {
        "de": "Typ",
        "en": "type",
    },
    "model_id": {
        "de": "Modell-ID",
        "en": "Model ID",
    },
    "author": {
        "de": "Autor",
        "en": "Author",
    },
    "pipeline": {
        "de": "Pipeline",
        "en": "Pipeline",
    },
    "params": {
        "de": "Parameter",
        "en": "Params",
    },
    "status": {
        "de": "Status",
        "en": "Status",
    },
    "notes": {
        "de": "Notizen",
        "en": "Notes",
    },
    "tier": {
        "de": "Stufe",
        "en": "Tier",
    },
}


def _l(key: str, lang: str = "de") -> str:
    """Get localized string for key in specified language."""
    strings = LOCALIZED_STRINGS.get(key, {})
    return strings.get(lang, strings.get("en", key))


def _get_bilingual_value(value: Any, lang: str, fallback_lang: str = "de") -> Any:
    """
    Extract value for specified language from bilingual field.
    
    If value is a dict with language keys, return the value for lang.
    If lang is not present, try fallback_lang.
    If neither, return the value as-is (legacy format).
    """
    if isinstance(value, dict) and ("de" in value or "en" in value):
        if lang in value:
            return value[lang]
        if fallback_lang in value:
            return value[fallback_lang]
        # Return first available
        for v in value.values():
            return v
    return value


class Reporter:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    @staticmethod
    def _escape_underscores(text: str) -> str:
        return text.replace("_", r"\_") if text else text

    @staticmethod
    def _format_params_b(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None

        if val <= 0:
            return "0M"

        if val < 1:
            rounded_b = math.floor(val * 100 + 0.5) / 100
            if rounded_b >= 1.0:
                # Handle values that round up to 1.0B (e.g. 0.995B -> 1.00B)
                return "1.0B"
            millions = int(round(rounded_b * 1000))
            return f"{millions}M"

        if val < 2:
            rounded = math.floor(val * 10 + 0.5) / 10
            if rounded >= 2.0:
                # If rounding pushes us to 2.0B, format as whole billions per requirements
                return f"{int(round(rounded))}B"
            return f"{rounded:.1f}B"

        return f"{int(round(val))}B"

    def write_markdown_report(
        self,
        stats: RunStats,
        processed_models: Optional[List[Dict[str, Any]]] = None,
        date_str: Optional[str] = None,
        language: str = "de",
        report_type: Literal["debug", "normal"] = "debug",
    ) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        ts = stats.started_at.astimezone(timezone.utc).strftime("%H%M%S")
        path = out / f"report_{date_str}_{ts}.md"

        lines: list[str] = []
        lines.append(f"# {_l('report_title', language)} ({date_str})")
        lines.append("")

        lines.append(f"## {_l('summary', language)}")
        lines.append(f"- {_l('discovered_unique', language)}: **{stats.candidates_total}**")
        lines.append(f"- {_l('queued_new_updated', language)}: **{stats.queued}**")
        lines.append(f"- {_l('noop_unchanged', language)}: **{stats.noop_unchanged}**")
        lines.append(f"- {_l('included_in_report', language)}: **{stats.processed}**")
        lines.append(f"- {_l('skipped', language)}: **{stats.skipped}**")
        lines.append(f"- {_l('warnings', language)}: **{stats.warned}**")
        lines.append(f"- {_l('llm_attempted', language)}: **{stats.llm_analyzed}**")
        lines.append(f"- {_l('llm_succeeded', language)}: **{stats.llm_succeeded}**")
        lines.append(f"- {_l('llm_failed', language)}: **{stats.llm_failed}**")
        lines.append("")

        # Debug sections: only show for debug report type
        if report_type == "debug":
            lines.append(f"## {_l('top_skip_reasons', language)}")
            if not stats.skip_reasons:
                lines.append(f"- {_l('none', language)}")
            else:
                for reason, cnt in stats.top_skip_reasons(20):
                    lines.append(f"- **{self._escape_underscores(reason)}**: {cnt}")

            lines.append("")
            lines.append(f"## {_l('top_warning_reasons', language)}")
            if not stats.warn_reasons:
                lines.append(f"- {_l('none', language)}")
            else:
                for reason, cnt in stats.top_warn_reasons(20):
                    lines.append(f"- **{self._escape_underscores(reason)}**: {cnt}")

            if stats.skip_reasons_by_uploader:
                lines.append("")
                lines.append(f"## {_l('top_skipped_uploaders', language)}")
                totals = Counter({u: sum(c.values()) for u, c in stats.skip_reasons_by_uploader.items()})
                for uploader, cnt in totals.most_common(15):
                    top3 = stats.skip_reasons_by_uploader[uploader].most_common(3)
                    top3_str = ", ".join([f"{self._escape_underscores(r)} ({n})" for r, n in top3]) if top3 else ""
                    lines.append(f"- **{self._escape_underscores(uploader)}**: {cnt}" + (f" — top: {top3_str}" if top3_str else ""))

            lines.append("")

        lines.append(f"## {_l('processed_models_overview', language)}")

        if not processed_models:
            lines.append(_l("no_models_criteria", language))
        else:
            included_uploaders = Counter((m.get("namespace") or m.get("author") or "unknown") for m in processed_models)
            lines.append("")
            lines.append(f"### {_l('top_included_uploaders', language)}")
            for uploader, cnt in included_uploaders.most_common(15):
                lines.append(f"- **{self._escape_underscores(uploader)}**: {cnt}")

            lines.append("")
            lines.append(f"### {_l('models', language)}")

            def _score(m: Dict[str, Any]) -> int:
                a = m.get("llm_analysis") or {}
                return int(a.get("specialist_score", 0) or 0)

            for m in sorted(processed_models, key=lambda x: (_score(x), x.get("id", "")), reverse=True)[:40]:
                mid = m.get("id", "")
                mid_display = self._escape_underscores(mid)
                uploader = m.get("namespace") or m.get("author") or "unknown"
                uploader_display = self._escape_underscores(uploader)
                a = m.get("llm_analysis") or {}
                score = a.get("specialist_score", 0)
                mtype = a.get("model_type", "N/A")
                link = f"https://huggingface.co/{mid}" if mid else ""
                if link:
                    lines.append(f"- **[{mid_display}]({link})** — {_l('uploader', language)}: **{uploader_display}** — {_l('score', language)}: **{score}** — {_l('type', language)}: {mtype}")
                else:
                    lines.append(f"- **{mid_display}** — {_l('uploader', language)}: **{uploader_display}** — {_l('score', language)}: **{score}** — {_l('type', language)}: {mtype}")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def generate_full_report(
        self,
        stats: RunStats,
        processed_models: List[Dict[str, Any]],
        date_str: Optional[str] = None,
        language: str = "de",
        report_type: Literal["debug", "normal"] = "debug",
    ) -> Path:
        path = self.write_markdown_report(
            stats,
            processed_models=processed_models,
            date_str=date_str,
            language=language,
            report_type=report_type,
        )
        base_content = path.read_text(encoding="utf-8")

        details: list[str] = []
        details.append("")
        details.append(f"## {_l('detailed_analysis', language)}")
        details.append("")

        if not processed_models:
            details.append(f"- {_l('no_processed_models', language)}")
        else:
            def _score(m: Dict[str, Any]) -> int:
                a = m.get("llm_analysis") or {}
                return int(a.get("specialist_score", 0) or 0)

            processed_models_sorted = sorted(processed_models, key=lambda x: (_score(x), x.get("id", "")), reverse=True)

            for m in processed_models_sorted:
                mid = m.get("id", "")
                name = m.get("name") or (mid.split("/")[-1] if mid else "unknown")
                name_display = self._escape_underscores(name)
                uploader = m.get("namespace") or m.get("author") or "unknown"
                uploader_display = self._escape_underscores(uploader)
                a = m.get("llm_analysis") or {}

                link = f"https://huggingface.co/{mid}" if mid else ""
                score = a.get("specialist_score", 0)
                mtype = a.get("model_type", "N/A")

                # Params: show total + active if available
                total_b = self._format_params_b(m.get("params_total_b"))
                active_b = self._format_params_b(m.get("params_active_b"))
                src = m.get("params_source") or "unknown"
                if total_b is None and active_b is None:
                    params_str = "Unknown"
                elif total_b is not None and active_b is not None:
                    params_str = f"total={total_b}, active={active_b} ({src})"
                else:
                    params_str = f"{total_b or active_b} ({src})"

                kind_str = m.get("author_kind", "unknown")
                tier_str = f"{_l('tier', language)} {m.get('trust_tier', '?')}"
                pipeline_tag = (m.get("pipeline_tag") or "unknown").lower()
                status = m.get("status", "processed")
                notes = m.get("report_notes", "")

                details.append(f"### [{name_display}]({link})" if link else f"### {name_display}")
                details.append("")
                details.append(
                    f"**{_l('model_id', language)}:** `{mid}`  \n"
                    f"**{_l('uploader', language).title()}:** **{uploader_display}**  \n"
                    f"**{_l('author', language)}:** {kind_str} ({tier_str})  \n"
                    f"**{_l('pipeline', language)}:** `{pipeline_tag}`  \n"
                    f"**{_l('type', language).title()}:** {mtype}  \n"
                    f"**{_l('score', language).title()}:** **{score}/10**  \n"
                    f"**{_l('params', language)}:** {params_str}  \n"
                    f"**{_l('status', language)}:** `{status}`"
                )
                if notes:
                    details.append(f"**{_l('notes', language)}:** {notes}")
                details.append("")

                # Get bilingual content for the selected language
                blurb = _get_bilingual_value(a.get("newsletter_blurb"), language)
                if blurb:
                    details.append(f"> {blurb}")
                    details.append("")

                key_facts = _get_bilingual_value(a.get("key_facts"), language) or []
                if key_facts:
                    details.append(f"#### {_l('key_facts', language)}")
                    for fact in key_facts:
                        details.append(f"- {fact}")
                    details.append("")

                delta = a.get("delta") or {}
                what_changed = _get_bilingual_value(delta.get("what_changed"), language) or []
                why_matters = _get_bilingual_value(delta.get("why_it_matters"), language) or []
                if what_changed or why_matters:
                    details.append(f"#### {_l('delta', language)}")
                    if what_changed:
                        details.append(f"**{_l('what_changed', language)}**")
                        details.append("")
                        for x in what_changed:
                            details.append(f"- {x}")
                    if why_matters:
                        details.append("")
                        details.append(f"**{_l('why_relevant', language)}**")
                        details.append("")
                        for x in why_matters:
                            details.append(f"- {x}")
                    details.append("")

                manu = a.get("manufacturing") or {}
                use_cases = _get_bilingual_value(manu.get("use_cases"), language) or []
                if use_cases:
                    details.append(f"#### {_l('manufacturing', language)}")
                    details.append(f"**{_l('use_cases', language)}**")
                    details.append("")
                    for x in use_cases:
                        details.append(f"- {x}")
                    details.append("")

                if config.REPORT_INCLUDE_EVIDENCE:
                    evidence = a.get("evidence") or []
                    if evidence:
                        details.append(f"#### {_l('evidence_quotes', language)}")
                        for e in evidence[:4]:
                            claim = (e.get("claim") or "").strip()
                            quote = (e.get("quote") or "").strip()
                            if claim:
                                details.append(f"- **{_l('claim', language)}:** {claim}")
                            if quote:
                                details.append(f'  - {_l("quote", language)}: "{quote}"')
                        details.append("")

                details.append("---")
                details.append("")

        full_content = base_content + "\n" + "\n".join(details)
        path.write_text(full_content, encoding="utf-8")
        return path

    def export_csv(self, models: List[Dict[str, Any]], filename: str = "labeling_pending.csv", language: str = "de") -> str:
        filepath = os.path.join(self.output_dir, filename)
        file_exists = os.path.exists(filepath)

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Modell-ID",
                    "Uploader",
                    "Typ",
                    "Score",
                    "Params_total_B",
                    "Params_active_B",
                    "Params_source",
                    "Blurb",
                    "Key Facts",
                    "Manufacturing Use Cases",
                ])

            for m in models:
                a = m.get("llm_analysis") or {}
                blurb = _get_bilingual_value(a.get("newsletter_blurb"), language) or ""
                key_facts = _get_bilingual_value(a.get("key_facts"), language) or []
                use_cases = _get_bilingual_value((a.get("manufacturing") or {}).get("use_cases"), language) or []
                writer.writerow([
                    m.get("id", ""),
                    m.get("namespace") or m.get("author") or "",
                    a.get("model_type", ""),
                    a.get("specialist_score", 0),
                    m.get("params_total_b"),
                    m.get("params_active_b"),
                    m.get("params_source"),
                    blurb if isinstance(blurb, str) else str(blurb),
                    " | ".join(key_facts) if isinstance(key_facts, list) else str(key_facts),
                    " | ".join(use_cases) if isinstance(use_cases, list) else str(use_cases),
                ])

        return filepath
