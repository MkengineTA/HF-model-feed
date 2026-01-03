# reporter.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from collections import Counter
import csv
import os

import config
from run_stats import RunStats

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

        if val < 1:
            rounded_b = round(val, 2)
            millions = int(rounded_b * 1000)
            return f"{millions}M"

        if val < 2:
            rounded = round(val, 1)
            return f"{rounded:.1f}B"

        return f"{int(round(val))}B"

    def write_markdown_report(
        self,
        stats: RunStats,
        processed_models: Optional[List[Dict[str, Any]]] = None,
        date_str: Optional[str] = None,
    ) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        ts = stats.started_at.astimezone(timezone.utc).strftime("%H%M%S")
        path = out / f"report_{date_str}_{ts}.md"

        lines: list[str] = []
        lines.append(f"# EdgeAIScout Report ({date_str})")
        lines.append("")

        lines.append("## Summary")
        lines.append(f"- Discovered (unique): **{stats.candidates_total}**")
        lines.append(f"- Queued (new/updated): **{stats.queued}**")
        lines.append(f"- No-op (unchanged/already tracked): **{stats.noop_unchanged}**")
        lines.append(f"- Included in report: **{stats.processed}**")
        lines.append(f"- Skipped: **{stats.skipped}**")
        lines.append(f"- Warnings: **{stats.warned}**")
        lines.append(f"- LLM attempted: **{stats.llm_analyzed}**")
        lines.append(f"- LLM succeeded: **{stats.llm_succeeded}**")
        lines.append(f"- LLM failed: **{stats.llm_failed}**")
        lines.append("")

        lines.append("## Top skip reasons")
        if not stats.skip_reasons:
            lines.append("- (none)")
        else:
            for reason, cnt in stats.top_skip_reasons(20):
                lines.append(f"- **{self._escape_underscores(reason)}**: {cnt}")

        lines.append("")
        lines.append("## Top warning reasons")
        if not stats.warn_reasons:
            lines.append("- (none)")
        else:
            for reason, cnt in stats.top_warn_reasons(20):
                lines.append(f"- **{self._escape_underscores(reason)}**: {cnt}")

        if stats.skip_reasons_by_uploader:
            lines.append("")
            lines.append("## Top skipped uploaders")
            totals = Counter({u: sum(c.values()) for u, c in stats.skip_reasons_by_uploader.items()})
            for uploader, cnt in totals.most_common(15):
                top3 = stats.skip_reasons_by_uploader[uploader].most_common(3)
                top3_str = ", ".join([f"{self._escape_underscores(r)} ({n})" for r, n in top3]) if top3 else ""
                lines.append(f"- **{self._escape_underscores(uploader)}**: {cnt}" + (f" — top: {top3_str}" if top3_str else ""))

        lines.append("")
        lines.append("## Processed models (overview)")

        if not processed_models:
            lines.append("No models met the criteria for this report.")
        else:
            included_uploaders = Counter((m.get("namespace") or m.get("author") or "unknown") for m in processed_models)
            lines.append("")
            lines.append("### Top included uploaders")
            for uploader, cnt in included_uploaders.most_common(15):
                lines.append(f"- **{self._escape_underscores(uploader)}**: {cnt}")

            lines.append("")
            lines.append("### Models")

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
                    lines.append(f"- **[{mid_display}]({link})** — uploader: **{uploader_display}** — score: **{score}** — type: {mtype}")
                else:
                    lines.append(f"- **{mid_display}** — uploader: **{uploader_display}** — score: **{score}** — type: {mtype}")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def generate_full_report(self, stats: RunStats, processed_models: List[Dict[str, Any]], date_str: Optional[str] = None) -> Path:
        path = self.write_markdown_report(stats, processed_models=processed_models, date_str=date_str)
        base_content = path.read_text(encoding="utf-8")

        details: list[str] = []
        details.append("")
        details.append("## Detailed analysis")
        details.append("")

        if not processed_models:
            details.append("- No processed models to display.")
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
                tier_str = f"Tier {m.get('trust_tier', '?')}"
                pipeline_tag = (m.get("pipeline_tag") or "unknown").lower()
                status = m.get("status", "processed")
                notes = m.get("report_notes", "")

                details.append(f"### [{name_display}]({link})" if link else f"### {name_display}")
                details.append("")
                details.append(
                    f"**Model ID:** `{mid}`  \n"
                    f"**Uploader:** **{uploader_display}**  \n"
                    f"**Author:** {kind_str} ({tier_str})  \n"
                    f"**Pipeline:** `{pipeline_tag}`  \n"
                    f"**Type:** {mtype}  \n"
                    f"**Score:** **{score}/10**  \n"
                    f"**Params:** {params_str}  \n"
                    f"**Status:** `{status}`"
                )
                if notes:
                    details.append(f"**Notes:** {notes}")
                details.append("")

                blurb = a.get("newsletter_blurb")
                if blurb:
                    details.append(f"> {blurb}")
                    details.append("")

                key_facts = a.get("key_facts") or []
                if key_facts:
                    details.append("#### Key facts")
                    for fact in key_facts:
                        details.append(f"- {fact}")
                    details.append("")

                delta = a.get("delta") or {}
                what_changed = delta.get("what_changed") or []
                why_matters = delta.get("why_it_matters") or []
                if what_changed or why_matters:
                    details.append("#### Delta")
                    if what_changed:
                        details.append("**Was ist neu?**")
                        details.append("")
                        for x in what_changed:
                            details.append(f"- {x}")
                    if why_matters:
                        details.append("")
                        details.append("**Warum relevant?**")
                        details.append("")
                        for x in why_matters:
                            details.append(f"- {x}")
                    details.append("")

                manu = a.get("manufacturing") or {}
                use_cases = manu.get("use_cases") or []
                if use_cases:
                    details.append("#### Manufacturing")
                    details.append("**Use cases**")
                    details.append("")
                    for x in use_cases:
                        details.append(f"- {x}")
                    details.append("")

                if config.REPORT_INCLUDE_EVIDENCE:
                    evidence = a.get("evidence") or []
                    if evidence:
                        details.append("#### Evidence (quotes)")
                        for e in evidence[:4]:
                            claim = (e.get("claim") or "").strip()
                            quote = (e.get("quote") or "").strip()
                            if claim:
                                details.append(f"- **Claim:** {claim}")
                            if quote:
                                details.append(f"  - Quote: “{quote}”")
                        details.append("")

                details.append("---")
                details.append("")

        full_content = base_content + "\n" + "\n".join(details)
        path.write_text(full_content, encoding="utf-8")
        return path

    def export_csv(self, models: List[Dict[str, Any]], filename: str = "labeling_pending.csv") -> str:
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
                writer.writerow([
                    m.get("id", ""),
                    m.get("namespace") or m.get("author") or "",
                    a.get("model_type", ""),
                    a.get("specialist_score", 0),
                    m.get("params_total_b"),
                    m.get("params_active_b"),
                    m.get("params_source"),
                    a.get("newsletter_blurb", ""),
                    " | ".join(a.get("key_facts", []) or []),
                    " | ".join((a.get("manufacturing", {}) or {}).get("use_cases", []) or []),
                ])

        return filepath
