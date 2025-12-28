import os
import sys
import logging
import argparse
from datetime import datetime
import dateutil.parser

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
    parser.add_argument("--limit", type=int, default=100, help="Limit for created/updated sources")
    parser.add_argument("--dry-run", action="store_true", help="Do not save to DB")
    args = parser.parse_args()

    # 1. Initialize Components
    db = Database(config.DB_PATH)
    hf_client = HFClient(token=config.HF_TOKEN)

    # Initialize LLM with OpenRouter support config
    llm_client = LLMClient(
        api_url=config.LLM_API_URL,
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        site_url=config.OR_SITE_URL,
        app_name=config.OR_APP_NAME
    )

    reporter = Reporter()

    # 2. Fetch Models from Sources
    logger.info("Fetching new models...")

    # Source A: Recently Created (Brand new)
    recent_models = hf_client.fetch_new_models(limit=args.limit)
    logger.info(f"Fetched {len(recent_models)} recently created models.")

    # Source B: Recently Updated (Existing but changed)
    updated_models = hf_client.fetch_recently_updated_models(limit=args.limit)
    logger.info(f"Fetched {len(updated_models)} recently updated models.")

    # Source C: Daily Papers (Science)
    paper_model_ids = hf_client.fetch_daily_papers(limit=20)
    logger.info(f"Fetched {len(paper_model_ids)} models from daily papers.")

    # 3. Combine Candidates & Determine Action

    # Map all candidates by ID
    candidates = {}

    # Helper to add/merge candidates
    def add_candidate(info, source_tag):
        if info.id not in candidates:
            candidates[info.id] = info
        # We could tag source if needed, but info.lastModified is key

    for m in recent_models: add_candidate(m, 'created')
    for m in updated_models: add_candidate(m, 'updated')

    # For paper models, we need to fetch info
    for mid in paper_model_ids:
        if mid not in candidates:
            info = hf_client.get_model_info(mid)
            if info:
                add_candidate(info, 'paper')

    logger.info(f"Total unique candidates found: {len(candidates)}")

    existing_ids = db.get_existing_ids()
    processing_queue = []

    for model_id, model_info in candidates.items():
        should_process = False
        reason = ""

        # Check logic:
        # 1. New ID -> Process
        if model_id not in existing_ids:
            should_process = True
            reason = "New Discovery"
        else:
            # 2. Existing ID -> Check Update
            # Get stored last_modified
            db_last_mod_str = db.get_model_last_modified(model_id)
            api_last_mod_dt = model_info.lastModified # datetime object from HfApi

            if db_last_mod_str and api_last_mod_dt:
                try:
                    # Ensure both are comparable (datetime)
                    # DB string usually ISO
                    db_last_mod_dt = dateutil.parser.parse(str(db_last_mod_str))

                    # Ensure timezone awareness match (usually UTC)
                    if api_last_mod_dt > db_last_mod_dt:
                        should_process = True
                        reason = f"Update Detected (API: {api_last_mod_dt} > DB: {db_last_mod_dt})"
                except Exception as e:
                    logger.warning(f"Date parsing error for {model_id}: {e}")
                    # If unsure, maybe skip or process? Let's skip to avoid loops, or process to fix?
                    # Safer to skip unless we are sure it's newer.
            elif not db_last_mod_str:
                # If we have it in DB but no timestamp (migration case), maybe re-process once?
                should_process = True
                reason = "Missing Timestamp in DB"

        if should_process:
            processing_queue.append((model_id, model_info, reason))

    logger.info(f"Models queued for analysis: {len(processing_queue)}")

    processed_models = []

    for model_id, model_info, reason in processing_queue:
        logger.info(f"Processing {model_id} ({reason})...")

        # --- Phase 0: Security & Deep Metadata ---
        file_details = hf_client.get_model_file_details(model_id)

        if not filters.is_secure(file_details):
             logger.warning(f"Skipping {model_id}: Security check failed")
             continue

        # --- Phase 1: Static Filters ---
        params_est = filters.extract_parameter_count(model_info, file_details)
        if params_est and params_est > config.MAX_PARAMS_BILLIONS:
            logger.info(f"Skipping {model_id}: Too large ({params_est:.2f}B)")
            continue

        if filters.is_quantized(model_id):
            logger.info(f"Skipping {model_id}: Quantized format")
            continue

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
            'last_modified': model_info.lastModified,
            'params_est': params_est,
            'hf_tags': model_info.tags,
            'llm_analysis': llm_result,
            'status': final_status
        }

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
