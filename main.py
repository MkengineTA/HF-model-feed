import os
import sys
import logging
import argparse
from datetime import datetime, timezone
import dateutil.parser

import config
from utils import setup_logging
from database import Database
from hf_client import HFClient
from llm_client import LLMClient
from reporter import Reporter
from mailer import Mailer
import filters

# Setup logging
logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Edge AI Scout & Specialist Model Monitor")
    parser.add_argument("--limit", type=int, default=1000, help="Safety limit for API fetching (default: 1000)")
    parser.add_argument("--dry-run", action="store_true", help="Do not save to DB")
    parser.add_argument("--force-email", action="store_true", help="Send email even in dry-run mode")
    args = parser.parse_args()

    current_run_time = datetime.now(timezone.utc)
    date_str = current_run_time.strftime("%Y-%m-%d")

    # 1. Initialize Components
    db = Database(config.DB_PATH)
    hf_client = HFClient(token=config.HF_TOKEN)

    llm_client = LLMClient(
        api_url=config.LLM_API_URL,
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        site_url=config.OR_SITE_URL,
        app_name=config.OR_APP_NAME,
        enable_reasoning=config.LLM_ENABLE_REASONING
    )

    reporter = Reporter()
    mailer = Mailer()

    last_run_ts = db.get_last_run_timestamp()
    logger.info(f"Last successful run: {last_run_ts}")

    # 2. Fetch Models
    logger.info("Fetching models...")
    recent_models = hf_client.fetch_new_models(since=last_run_ts, limit=args.limit)
    logger.info(f"Fetched {len(recent_models)} recently created models since last run.")
    updated_models = hf_client.fetch_recently_updated_models(since=last_run_ts, limit=args.limit)
    logger.info(f"Fetched {len(updated_models)} recently updated models since last run.")
    trending_ids = hf_client.fetch_trending_models(limit=10)
    logger.info(f"Fetched {len(trending_ids)} trending models.")
    paper_model_ids = hf_client.fetch_daily_papers(limit=20)
    logger.info(f"Fetched {len(paper_model_ids)} models from daily papers.")

    # 3. Combine Candidates
    candidates = {}
    def add_candidate(info, source_tag):
        if info.id not in candidates:
            candidates[info.id] = info

    for m in recent_models: add_candidate(m, 'created')
    for m in updated_models: add_candidate(m, 'updated')

    extra_ids = set(trending_ids + paper_model_ids)
    for mid in extra_ids:
        if mid not in candidates:
            info = hf_client.get_model_info(mid)
            if info:
                add_candidate(info, 'trending_or_paper')

    logger.info(f"Total unique candidates found: {len(candidates)}")

    existing_ids = db.get_existing_ids()
    processing_queue = []

    for model_id, model_info in candidates.items():
        should_process = False
        reason = ""

        if model_id not in existing_ids:
            should_process = True
            reason = "New Discovery"
        else:
            db_last_mod_str = db.get_model_last_modified(model_id)
            api_last_mod_dt = model_info.lastModified
            if db_last_mod_str and api_last_mod_dt:
                try:
                    db_last_mod_dt = dateutil.parser.parse(str(db_last_mod_str))
                    if api_last_mod_dt > db_last_mod_dt:
                        should_process = True
                        reason = f"Update Detected"
                except Exception:
                    pass
            elif not db_last_mod_str:
                should_process = True
                reason = "Missing Timestamp"

        if should_process:
            processing_queue.append((model_id, model_info, reason))

    logger.info(f"Models queued for analysis: {len(processing_queue)}")

    processed_models = []

    for model_id, model_info, reason in processing_queue:
        logger.info(f"Processing {model_id} ({reason})...")

        # --- Phase 0: Quick Metadata Filters (Cheap) ---
        tags = model_info.tags or []

        if filters.is_excluded_content(model_id, tags):
            logger.info(f"Skipping {model_id}: Excluded content")
            continue

        if filters.is_generative_visual(model_info, tags):
            logger.info(f"Skipping {model_id}: Generative/3D/Diffusion")
            continue

        # Only fetch file details if passed metadata filters
        file_details = hf_client.get_model_file_details(model_id)

        if not filters.is_secure(file_details):
             logger.warning(f"Skipping {model_id}: Security check failed")
             continue

        # --- Phase 1: Deep Filters ---
        if filters.is_quantized(model_id, tags, file_details):
            logger.info(f"Skipping {model_id}: Quantized/Export format")
            continue

        params_est = filters.extract_parameter_count(model_info, file_details)
        if params_est and params_est > config.MAX_PARAMS_BILLIONS:
            logger.info(f"Skipping {model_id}: Too large ({params_est:.2f}B)")
            continue

        # --- Phase 2: README Validation ---
        readme_content = hf_client.get_model_readme(model_id)
        if not readme_content:
            logger.info(f"Skipping {model_id}: No README")
            continue

        if filters.is_merge(model_id, readme_content):
            logger.info(f"Skipping {model_id}: Merge model")
            continue

        # Advanced Robotics Filter (with VQA override)
        if filters.is_robotics_or_vla(model_info, tags, readme_content):
            logger.info(f"Skipping {model_id}: Robotics/Embodied (No Inspection overlap)")
            continue

        if filters.is_boilerplate_readme(readme_content):
            # If inspection pipeline, maybe keep review_required?
            # For now, stick to strict quality
            logger.info(f"Skipping {model_id}: Boilerplate/Empty README")
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
            llm_result = llm_client.analyze_model(readme_content, tags)

            # Post-Analysis Validation (Evidence Check)
            if llm_result:
                evidence = llm_result.get('evidence', [])
                valid_evidence = True
                if evidence:
                    for item in evidence:
                        quote = item.get('quote', '')
                        if quote and quote not in readme_content:
                            logger.warning(f"Evidence validation failed for {model_id}. Quote not found: {quote[:50]}...")
                            # Don't discard, but maybe flag confidence?
                            # Or strict mode: set status to review
                            # For now, let's just log.

            if not llm_result:
                logger.error(f"LLM analysis failed for {model_id}")
                final_status = 'error'
            else:
                logger.info(f"Analysis complete for {model_id}: Score {llm_result.get('specialist_score')}")

        model_data = {
            'id': model_id,
            'name': model_id.split('/')[-1],
            'author': model_id.split('/')[0] if '/' in model_id else '',
            'created_at': model_info.created_at,
            'last_modified': model_info.lastModified,
            'params_est': params_est,
            'hf_tags': tags,
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
        md_path = reporter.generate_markdown_report(processed_models, date_str=date_str)
        reporter.export_csv(processed_models)
        logger.info(f"Report generated: {md_path}")

        if not args.dry_run or args.force_email:
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                mailer.send_report(md_content, date_str)
            except Exception as e:
                logger.error(f"Failed to read report for email dispatch: {e}")
        else:
            logger.info("Email dispatch skipped (Dry-Run active).")

    else:
        logger.info("No new models processed.")

    if not args.dry_run:
        db.set_last_run_timestamp(current_run_time)
        logger.info(f"Updated last run timestamp to {current_run_time}")
    else:
        logger.info("State update skipped (Dry-Run active).")

if __name__ == "__main__":
    main()
