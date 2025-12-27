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

    # 2. Fetch Models from Multiple Sources
    logger.info("Fetching new models...")

    # Source A: Newest
    recent_models = hf_client.fetch_new_models(limit=args.limit)
    logger.info(f"Fetched {len(recent_models)} recent models.")

    # Source B: Trending
    trending_ids = hf_client.fetch_trending_models(limit=10)
    logger.info(f"Fetched {len(trending_ids)} trending models.")

    # Source C: Daily Papers
    paper_model_ids = hf_client.fetch_daily_papers(limit=20)
    logger.info(f"Fetched {len(paper_model_ids)} models from daily papers.")

    # 3. Combine and Deduplicate
    existing_ids = db.get_existing_ids()

    # Convert recent_models to a dictionary for easy access, keyed by ID
    candidates = {m.id: m for m in recent_models}

    # Add trending and paper models (need to fetch info if not already in recent)
    extra_ids = set(trending_ids + paper_model_ids)
    for mid in extra_ids:
        if mid not in candidates:
            # We need to fetch info for these
            info = hf_client.get_model_info(mid)
            if info:
                candidates[mid] = info

    # Deduplicate against DB
    new_model_ids = [mid for mid in candidates.keys() if mid not in existing_ids]
    logger.info(f"New candidates after deduplication: {len(new_model_ids)}")

    processed_models = []

    for model_id in new_model_ids:
        logger.info(f"Processing {model_id}...")
        model_info = candidates[model_id]

        # --- Phase 0: Security & Deep Metadata ---
        # Fetch file details for Security Check and Param Estimation
        file_details = hf_client.get_model_file_details(model_id)

        # Security Check
        if not filters.is_secure(file_details):
             logger.warning(f"Skipping {model_id}: Security check failed (Unsafe/Malware detected)")
             continue

        # --- Phase 1: Static Filters ---

        # Parameter Count
        params_est = filters.extract_parameter_count(model_info, file_details)
        if params_est and params_est > config.MAX_PARAMS_BILLIONS:
            logger.info(f"Skipping {model_id}: Too large ({params_est:.2f}B)")
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
