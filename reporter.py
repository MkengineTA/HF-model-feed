from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from run_stats import RunStats
from collections import Counter
import csv
import os

class Reporter:
    def __init__(self, output_dir="."):
        self.output_dir = output_dir

    def write_markdown_report(
        self,
        stats: RunStats,
        date_str: str = None,
    ) -> Path:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # We append timestamp to filename to avoid overwrites if running multiple times/day
        ts = stats.started_at.astimezone(timezone.utc).strftime("%H%M%S")
        path = out / f"report_{date_str}_{ts}.md"

        lines: list[str] = []
        lines.append(f"# EdgeAIScout Report ({date_str})")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- Candidates: **{stats.candidates_total}**")
        lines.append(f"- Processed: **{stats.processed}**")
        lines.append(f"- Skipped: **{stats.skipped}**")
        lines.append(f"- LLM analyzed: **{stats.llm_analyzed}**")
        lines.append("")

        lines.append("## Top skip reasons")
        if not stats.skip_reasons:
            lines.append("- (none)")
        else:
            for reason, cnt in stats.top_skip_reasons(20):
                lines.append(f"- **{reason}**: {cnt}")

        # Top Skipped Uploaders
        uploader_skips = Counter(item.author for item in stats.skip_items if item.author)
        if uploader_skips:
            lines.append("")
            lines.append("## Top Skipped Uploaders")
            for uploader, cnt in uploader_skips.most_common(10):
                lines.append(f"- **{uploader}**: {cnt}")

        lines.append("")
        lines.append("## Processed Models")
        if not stats.processed_items:
            lines.append("No models met the criteria for this report.")
        else:
            # We need to fetch details? RunStats only has ModelRef (id, uploader).
            # But main.py has `processed_models` list with full data.
            # Reporter.generate_markdown_report was getting full list.
            # I should keep `generate_markdown_report` logic but integrate RunStats?
            # Or `write_markdown_report` assumes `stats` has ref, but we want full content.
            # I will modify `write_markdown_report` to accept `full_models_list` as well.
            # But the plan implies `write_markdown_report` replaces `generate_markdown_report`.
            pass

        # Since main.py collects `processed_models` (full dicts), I will pass that list
        # to a helper or just append it here if passed.
        # But `RunStats` only stores Refs.
        # I will change signature to accept `processed_models_data` separately.

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def generate_full_report(self, stats: RunStats, processed_models: list, date_str=None):
        # This combines stats and detailed model cards
        path = self.write_markdown_report(stats, date_str)

        # Read base stats
        base_content = path.read_text(encoding="utf-8")

        # Append Model Cards
        details = []
        details.append("\n## Detailed Analysis\n")

        if not processed_models:
            details.append("- No processed models to display.")
        else:
            # Sort by score
            processed_models.sort(key=lambda x: x['llm_analysis'].get('specialist_score', 0), reverse=True)

            for m in processed_models:
                a = m['llm_analysis']
                name = m['name']
                name_disp = f"`{name}`"
                link = f"https://huggingface.co/{m['id']}"
                params = a.get('params_m')
                params_str = f"{params}M" if params else "Unknown"

                tier_str = f"Tier {m.get('trust_tier', '?')}"
                kind_str = m.get('author_kind', 'unknown')

                details.append(f"### [{name_disp}]({link})")
                details.append(f"**Typ:** {a.get('model_type', 'N/A')} | **Score:** {a.get('specialist_score', 0)}/10 | **Params:** {params_str} | **Author:** {kind_str} ({tier_str})\n")
                details.append(f"> {a.get('newsletter_blurb', 'N/A')}\n")

                details.append("#### 🔑 Key Facts")
                for fact in a.get('key_facts', []):
                    details.append(f"- {fact}")

                delta = a.get('delta', {})
                if delta.get('what_changed'):
                    details.append("\n#### 🔺 Delta")
                    details.append("**Was ist neu?**\n")
                    for item in delta.get('what_changed', []):
                        details.append(f"- {item}")

                manu = a.get('manufacturing', {})
                if manu.get('use_cases'):
                    details.append("\n#### 🏭 Manufacturing Fit")
                    details.append("**Use Cases:**\n")
                    for item in manu.get('use_cases', []):
                        details.append(f"- {item}")

                confidence = a.get('confidence', 'low')
                conf_icon = "🟢" if confidence == 'high' else "🟡" if confidence == 'medium' else "🔴"
                unknowns = a.get('unknowns', [])
                unknowns_str = ', '.join(unknowns) if unknowns else "None"
                details.append(f"\n_{conf_icon} Confidence: {confidence} | Unknowns: {unknowns_str}_\n")
                details.append("---\n")

        full_content = base_content + "\n" + "\n".join(details)
        path.write_text(full_content, encoding="utf-8")
        return path

    def export_csv(self, models, filename="labeling_pending.csv"):
        filepath = os.path.join(self.output_dir, filename)
        file_exists = os.path.exists(filepath)

        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Modell-ID", "Typ", "Score", "Blurb", "Key Facts", "Manufacturing Use Cases"])

            for m in models:
                a = m.get('llm_analysis')
                if a:
                    writer.writerow([
                        m['id'],
                        a.get('model_type', ''),
                        a.get('specialist_score', 0),
                        a.get('newsletter_blurb', ''),
                        " | ".join(a.get('key_facts', [])),
                        " | ".join(a.get('manufacturing', {}).get('use_cases', []))
                    ])
