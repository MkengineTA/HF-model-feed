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

        high_potential = []
        review_required = []

        for m in models:
            analysis = m.get('llm_analysis')

            if analysis:
                score = analysis.get('specialist_score', 0)
                if score >= 7:
                    high_potential.append(m)
            elif m.get('status') == 'review_required':
                 review_required.append(m)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Daily Edge AI Scout Report - {date_str}\n\n")
            f.write(f"**Gescannte Modelle:** {len(models)}\n\n")

            f.write("## 🚀 High Potential (Specialist Score > 7)\n\n")
            if high_potential:
                for m in high_potential:
                    analysis = m['llm_analysis']
                    f.write(f"### [{m['name']}](https://huggingface.co/{m['id']})\n")
                    f.write(f"- **Kategorie:** {analysis.get('category', 'N/A')}\n")
                    f.write(f"- **USP:** {analysis.get('summary', 'N/A')}\n")
                    f.write(f"- **Tags:** {', '.join(m.get('hf_tags', [])[:5])}...\n\n")
            else:
                f.write("Keine High-Potential Modelle heute gefunden.\n\n")

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
                # German headers as requested
                writer.writerow(["Modell-ID", "Kategorie", "Zusammenfassung", "User-Label"])

            for m in models:
                analysis = m.get('llm_analysis')
                if analysis:
                    writer.writerow([
                        m['id'],
                        analysis.get('category', 'Unknown'),
                        analysis.get('summary', ''),
                        "" # user_label empty
                    ])
