import csv
import os
from datetime import datetime

class Reporter:
    def __init__(self, output_dir="."):
        self.output_dir = output_dir

    def generate_markdown_report(self, models, date_str=None):
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        filename = os.path.join(self.output_dir, f"report_{date_str}.md")

        processed_list = []
        review_required = []

        for m in models:
            analysis = m.get('llm_analysis')

            if analysis:
                processed_list.append(m)
            elif m.get('status') == 'review_required':
                 review_required.append(m)

        # Sort processed list by specialist_score DESC
        processed_list.sort(key=lambda x: x['llm_analysis'].get('specialist_score', 0), reverse=True)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Edge AI Scout Report - {date_str}\n\n")
            f.write(f"**Gescannte Modelle:** {len(models)}\n\n")

            if processed_list:
                for m in processed_list:
                    a = m['llm_analysis']

                    # Fields
                    name = m['name']
                    link = f"https://huggingface.co/{m['id']}"

                    # Metadata Header
                    f.write(f"## [{name}]({link})\n")
                    f.write(f"**Typ:** {a.get('model_type', 'N/A')} | ")
                    f.write(f"**Score:** {a.get('specialist_score', 0)}/10 | ")
                    f.write(f"**Params:** {a.get('params_m', 'N/A')}M\n\n")

                    # Blurb
                    f.write(f"> {a.get('newsletter_blurb', 'Keine Beschreibung verfügbar.')}\n\n")

                    # Key Facts
                    f.write("### 🔑 Key Facts\n")
                    for fact in a.get('key_facts', []):
                        f.write(f"- {fact}\n")
                    f.write("\n")

                    # Delta (Nested)
                    delta = a.get('delta', {})
                    f.write("### 🔺 Delta (vs. Base)\n")
                    if delta.get('what_changed'):
                        f.write("**Was ist neu?**\n")
                        for item in delta.get('what_changed', []):
                            f.write(f"- {item}\n")
                    if delta.get('why_it_matters'):
                        f.write("**Warum wichtig?**\n")
                        for item in delta.get('why_it_matters', []):
                            f.write(f"- {item}\n")
                    f.write("\n")

                    # Manufacturing
                    manu = a.get('manufacturing', {})
                    f.write("### 🏭 Manufacturing Fit\n")
                    if manu.get('use_cases'):
                        f.write("**Use Cases:**\n")
                        for item in manu.get('use_cases', []):
                            f.write(f"- {item}\n")
                    if manu.get('risks'):
                        f.write("**Risiken:**\n")
                        for item in manu.get('risks', []):
                            f.write(f"- {item}\n")
                    f.write("\n")

                    # Edge
                    edge = a.get('edge', {})
                    f.write("### ⚡ Edge Readiness\n")
                    f.write(f"- **Edge Ready:** {'✅' if edge.get('edge_ready') else '❌'}\n")
                    if edge.get('min_vram_gb'):
                        f.write(f"- **Min VRAM:** {edge.get('min_vram_gb')} GB\n")
                    if edge.get('deployment_notes'):
                        for item in edge.get('deployment_notes', []):
                            f.write(f"- {item}\n")
                    f.write("\n")

                    # Footer
                    confidence = a.get('confidence', 'low')
                    conf_icon = "🟢" if confidence == 'high' else "🟡" if confidence == 'medium' else "🔴"
                    f.write(f"_{conf_icon} Confidence: {confidence} | Unknowns: {', '.join(a.get('unknowns', []))}_\n")

                    f.write("---\n\n")
            else:
                f.write("Keine Modelle erfolgreich analysiert.\n\n")

            f.write("## 🔍 Review Required (Dünne Doku, Externe Links)\n\n")
            if review_required:
                for m in review_required:
                    f.write(f"- [{m['name']}](https://huggingface.co/{m['id']}) - Bitte manuell prüfen.\n")
            else:
                f.write("Keine Modelle für manuellen Review.\n\n")

        return filename

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
