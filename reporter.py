from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from collections import Counter
import csv
import os

from run_stats import RunStats


class Reporter:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    @staticmethod
    def _escape_underscores(text: str) -> str:
        """
        Escapes underscores so model names render correctly in Markdown.
        """
        if not text:
            return text
        return text.replace("_", r"\_")

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

        # --- Summary ---
        lines.append("## Summary")
        lines.append(f"- Candidates: **{stats.candidates_total}**")
        lines.append(f"- Processed (kept for report): **{stats.processed}**")
        lines.append(f"- Skipped: **{stats.skipped}**")
        lines.append(f"- LLM analyzed: **{stats.llm_analyzed}**")
        lines.append("")

        # --- Skip reasons ---
        lines.append("## Top skip reasons")
        if not stats.skip_reasons:
            lines.append("- (none)")
        else:
            for reason, cnt in stats.top_skip_reasons(20):
                lines.append(f"- **{reason}**: {cnt}")

        # --- Top skipped uploaders ---
        uploader_skips = Counter(item.author for item in stats.skip_items if item.author)
        if uploader_skips:
            lines.append("")
            lines.append("## Top skipped uploaders")
            for uploader, cnt in uploader_skips.most_common(15):
                lines.append(f"- **{uploader}**: {cnt}")

        # --- Processed models overview ---
        lines.append("")
        lines.append("## Processed models (overview)")

        if not processed_models:
            lines.append("No models met the criteria for this report.")
        else:
            # Top included uploaders
            included_uploaders = Counter((m.get("namespace") or m.get("author") or "unknown") for m in processed_models)
            lines.append("")
            lines.append("### Top included uploaders")
            for uploader, cnt in included_uploaders.most_common(15):
                lines.append(f"- **{uploader}**: {cnt}")

            # Short table-ish list (top N)
            lines.append("")
            lines.append("### Models")
            # sort by score desc, then id
            def _score(m: Dict[str, Any]) -> int:
                a = m.get("llm_analysis") or {}
                return int(a.get("specialist_score", 0) or 0)

            for m in sorted(processed_models, key=lambda x: (_score(x), x.get("id", "")), reverse=True)[:40]:
                mid = m.get("id", "")
                mid_display = self._escape_underscores(mid)
                uploader = m.get("namespace") or m.get("author") or "unknown"
                a = m.get("llm_analysis") or {}
                score = a.get("specialist_score", 0)
                mtype = a.get("model_type", "N/A")
                link = f"https://huggingface.co/{mid}" if mid else ""
                if link:
                    lines.append(f"- **[{mid_display}]({link})** — uploader: **{uploader}** — score: **{score}** — type: {mtype}")
                else:
                    lines.append(f"- **{mid_display}** — uploader: **{uploader}** — score: **{score}** — type: {mtype}")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def generate_full_report(
        self,
        stats: RunStats,
        processed_models: List[Dict[str, Any]],
        date_str: Optional[str] = None
    ) -> Path:
        """
        Writes a base report (stats + overview) and appends detailed model cards.
        """
        path = self.write_markdown_report(stats, processed_models=processed_models, date_str=date_str)
        base_content = path.read_text(encoding="utf-8")

        details: list[str] = []
        details.append("")
        details.append("## Detailed analysis")
        details.append("")

        if not processed_models:
            details.append("- No processed models to display.")
        else:
            # Sort by score desc
            def _score(m: Dict[str, Any]) -> int:
                a = m.get("llm_analysis") or {}
                return int(a.get("specialist_score", 0) or 0)

            processed_models_sorted = sorted(
                processed_models,
                key=lambda x: (_score(x), x.get("id", "")),
                reverse=True
            )

            for m in processed_models_sorted:
                mid = m.get("id", "")
                name = m.get("name") or (mid.split("/")[-1] if mid else "unknown")
                name_display = self._escape_underscores(name)
                uploader = m.get("namespace") or m.get("author") or "unknown"
                a = m.get("llm_analysis") or {}

                link = f"https://huggingface.co/{mid}" if mid else ""
                score = a.get("specialist_score", 0)
                mtype = a.get("model_type", "N/A")
                params = a.get("params_m")
                params_str = f"{params}M" if params else "Unknown"
                kind_str = m.get("author_kind", "unknown")
                tier_str = f"Tier {m.get('trust_tier', '?')}"
                pipeline_tag = (m.get("pipeline_tag") or "unknown").lower()
                status = m.get("status", "processed")
                notes = m.get("report_notes", "")

                details.append(f"### [{name_display}]({link})" if link else f"### {name_display}")
                details.append("")
                details.append(
                    f"**Model ID:** `{mid}`  \n"
                    f"**Uploader:** **{uploader}**  \n"
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
                        for x in what_changed:
                            details.append(f"- {x}")
                    if why_matters:
                        details.append("")
                        details.append("**Warum relevant?**")
                        for x in why_matters:
                            details.append(f"- {x}")
                    details.append("")

                edge = a.get("edge") or {}
                dep_notes = edge.get("deployment_notes") or []
                if dep_notes:
                    details.append("#### Deployment notes")
                    for x in dep_notes:
                        details.append(f"- {x}")
                    details.append("")

                manu = a.get("manufacturing") or {}
                use_cases = manu.get("use_cases") or []
                risks = manu.get("risks") or []
                if use_cases or risks:
                    details.append("#### Manufacturing")
                    if use_cases:
                        details.append("**Use cases**")
                        for x in use_cases:
                            details.append(f"- {x}")
                    if risks:
                        details.append("")
                        details.append("**Risks**")
                        for x in risks:
                            details.append(f"- {x}")
                    details.append("")

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

                unknowns = a.get("unknowns") or []
                confidence = a.get("confidence", "low")
                if unknowns:
                    details.append(f"_Confidence: **{confidence}** | Unknowns: {', '.join(unknowns)}_")
                else:
                    details.append(f"_Confidence: **{confidence}**_")
                details.append("")
                details.append("---")
                details.append("")

        full_content = base_content + "\n" + "\n".join(details)
        path.write_text(full_content, encoding="utf-8")
        return path

    def export_csv(self, models: List[Dict[str, Any]], filename: str = "labeling_pending.csv") -> str:
        """
        Appends to CSV (keeps previous behavior), but now includes uploader.
        """
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
                    "Blurb",
                    "Key Facts",
                    "Manufacturing Use Cases"
                ])

            for m in models:
                a = m.get("llm_analysis") or {}
                writer.writerow([
                    m.get("id", ""),
                    m.get("namespace") or m.get("author") or "",
                    a.get("model_type", ""),
                    a.get("specialist_score", 0),
                    a.get("newsletter_blurb", ""),
                    " | ".join(a.get("key_facts", []) or []),
                    " | ".join((a.get("manufacturing", {}) or {}).get("use_cases", []) or []),
                ])

        return filepath
