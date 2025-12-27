import os
import sys
import logging
import argparse
from datetime import datetime

import config
from utils import setup_logging
from database import Database
from hf_client import HFClient
from llm_client import LLMClient
from reporter import Reporter
import filters

# Setup logging
logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Edge AI Scout & Specialist Model Monitor")
    parser.add_argument("--limit", type=int, default=200, help="Number of models to scan")
    parser.add_argument("--dry-run", action="store_true", help="Do not save to DB")
    args = parser.parse_args()

    # 1. Initialize Components
    db = Database(config.DB_PATH)
    hf_client = HFClient(token=config.HF_TOKEN)
    llm_client = LLMClient(api_url=config.LLM_API_URL, model=config.LLM_MODEL)
    reporter = Reporter()

    # 2. Fetch Models
    logger.info(f"Fetching last {args.limit} models from Hugging Face...")
    recent_models = hf_client.fetch_new_models(limit=args.limit)
    logger.info(f"Fetched {len(recent_models)} models.")

    # 3. Deduplication
    existing_ids = db.get_existing_ids()
    new_models_candidates = [m for m in recent_models if m.id not in existing_ids]
    logger.info(f"New candidates after deduplication: {len(new_models_candidates)}")

    processed_models = []

    for model_info in new_models_candidates:
        model_id = model_info.id
        logger.info(f"Processing {model_id}...")

        # --- Phase 1: Static Filters ---

        # Parameter Count
        params_est = filters.extract_parameter_count(model_info)
        # If params_est is None, we assume it's okay/small enough to check or we default to keep?
        # Safe strategy: If we can't determine, we keep it for now (or maybe check later).
        # But specification says: "Modelle mit >10B Parametern ausschließen."
        if params_est and params_est > config.MAX_PARAMS_BILLIONS:
            logger.info(f"Skipping {model_id}: Too large ({params_est}B)")
            continue

        # Quantization
        if filters.is_quantized(model_id):
            logger.info(f"Skipping {model_id}: Quantized format")
            continue

        # Excluded Content
        if filters.is_excluded_content(model_id, model_info.tags):
            logger.info(f"Skipping {model_id}: Excluded content/tags")
            continue

        # --- Phase 2: README Validation ---
        readme_content = hf_client.get_model_readme(model_id)
        if not readme_content:
            logger.info(f"Skipping {model_id}: No README")
            continue

        readme_len = len(readme_content)
        final_status = 'processed'
        llm_result = None

        if readme_len < 300:
            if filters.has_external_links(readme_content):
                logger.info(f"{model_id}: Short README but has links -> Review Required")
                final_status = 'review_required'
            else:
                logger.info(f"Skipping {model_id}: Short README ({readme_len} chars) and no links")
                continue
        else:
            # --- Phase 3: LLM Analysis ---
            logger.info(f"Analyzing {model_id} with LLM...")
            llm_result = llm_client.analyze_model(readme_content, model_info.tags)
            if not llm_result:
                # If LLM fails, we log error but maybe don't save or save as error?
                # Spec: "Bei Absturz des lokalen LLM ... wird der Datenbank-Eintrag ... nicht erstellt oder auf error gesetzt"
                logger.error(f"LLM analysis failed for {model_id}")
                final_status = 'error'
            else:
                logger.info(f"Analysis complete for {model_id}: Score {llm_result.get('specialist_score')}")

        # Prepare Data Object
        model_data = {
            'id': model_id,
            'name': model_id.split('/')[-1],
            'author': model_id.split('/')[0] if '/' in model_id else '',
            'created_at': model_info.created_at,
            'params_est': params_est,
            'hf_tags': model_info.tags,
            'llm_analysis': llm_result,
            'status': final_status
        }

        # Don't save errors to DB so they are retried next time (as they won't be in existing_ids)
        # OR save as 'error' and handle retry logic?
        # Spec says: "nicht erstellt oder auf error gesetzt, damit es im nächsten Nachtlauf erneut geprüft wird."
        # If we save as 'error', next run `get_existing_ids` includes it, so we filter it out.
        # So we should NOT save it if we want retry, OR we update `get_existing_ids` to exclude 'error' status.
        # Let's adjust `get_existing_ids` in DB or just not save here.
        # Simpler: Don't save if status is error.

        if final_status != 'error':
            processed_models.append(model_data)
            if not args.dry_run:
                db.save_model(model_data)

    # 4. Generate Output
    if processed_models:
        logger.info(f"Generating reports for {len(processed_models)} processed models...")
        md_path = reporter.generate_markdown_report(processed_models)
        reporter.export_csv(processed_models)
        logger.info(f"Report generated: {md_path}")
    else:
        logger.info("No new models processed.")

if __name__ == "__main__":
    main()
