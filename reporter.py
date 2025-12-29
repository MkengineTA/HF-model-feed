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
            f.write(f"# Daily Edge AI Scout Report - {date_str}\n\n")
            f.write(f"**Gescannte Modelle:** {len(models)}\n\n")

            f.write("## 📋 Processed Models (Sortiert nach Specialist Score)\n\n")

            if processed_list:
                for m in processed_list:
                    analysis = m['llm_analysis']

                    # Fields
                    name = m['name']
                    link = f"https://huggingface.co/{m['id']}"
                    m_type = analysis.get('model_type', 'N/A')
                    base_model = analysis.get('base_model')
                    basis_str = base_model if base_model else "N/A"
                    summary = analysis.get('technical_summary', 'N/A')
                    delta = analysis.get('delta_explanation', 'N/A')
                    score = analysis.get('specialist_score', 0)

                    # Tag formatting: Comma separated, no hashtags
                    tags = m.get('hf_tags', [])
                    tags_str = ', '.join(tags[:10])

                    f.write(f"### [{name}]({link})\n")
                    f.write(f"- **Score:** {score}/10\n")
                    f.write(f"- **Typ:** {m_type}\n")
                    f.write(f"- **Basis:** {basis_str}\n")
                    f.write(f"- **Zusammenfassung:**\n{summary}\n") # Newline before content for lists
                    f.write(f"- **Das Delta:**\n{delta}\n") # Newline before content for lists
                    f.write(f"- **Tags:** {tags_str}\n")
                    f.write(f"- **Daten-Quelle:** README / Metadaten-Inferenz\n\n")
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
                # German headers
                writer.writerow(["Modell-ID", "Typ", "Basis", "Kategorie", "Zusammenfassung", "Delta", "User-Label"])

            for m in models:
                analysis = m.get('llm_analysis')
                if analysis:
                    writer.writerow([
                        m['id'],
                        analysis.get('model_type', ''),
                        analysis.get('base_model', ''),
                        analysis.get('category', 'Unknown'),
                        analysis.get('technical_summary', ''),
                        analysis.get('delta_explanation', ''),
                        "" # user_label empty
                    ])
