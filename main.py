import os
import sys
import logging
import argparse
from datetime import datetime, timezone
import dateutil.parser
import yaml
import re

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

def extract_yaml_front_matter(readme_text):
    if not readme_text:
        return None
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', readme_text, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Edge AI Scout & Specialist Model Monitor")
    parser.add_argument("--limit", type=int, default=1000, help="Safety limit for API fetching (default: 1000)")
    parser.add_argument("--dry-run", action="store_true", help="Do not save to DB")
    parser.add_argument("--force-email", action="store_true", help="Send email even in dry-run mode")
    args = parser.parse_args()

    current_run_time = datetime.now(timezone.utc)
    date_str = current_run_time.strftime("%Y-%m-%d")

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

    # Fetching
    logger.info("Fetching models...")
    recent_models = hf_client.fetch_new_models(since=last_run_ts, limit=args.limit)
    updated_models = hf_client.fetch_recently_updated_models(since=last_run_ts, limit=args.limit)
    trending_ids = hf_client.fetch_trending_models(limit=10)
    paper_model_ids = hf_client.fetch_daily_papers(limit=20)

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

    logger.info(f"Total unique candidates: {len(candidates)}")
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
                        reason = "Update Detected"
                except Exception:
                    pass
            elif not db_last_mod_str:
                should_process = True
                reason = "Missing Timestamp"

        if should_process:
            processing_queue.append((model_id, model_info, reason))

    processed_models = []

    for model_id, model_info, reason in processing_queue:
        logger.info(f"Processing {model_id} ({reason})...")

        # --- Author / Namespace Logic ---
        namespace = model_id.split('/')[0] if '/' in model_id else None
        trust_tier = 1 # Default: Normal User
        author_kind = 'unknown'

        if namespace:
            auth_entry = db.get_author(namespace)

            cache_valid = False
            if auth_entry:
                last_checked = dateutil.parser.parse(str(auth_entry['last_checked']))
                if (datetime.now(timezone.utc) - last_checked).days < 14:
                    cache_valid = True

            auth_data = None
            if not cache_valid:
                logger.info(f"Validating author: {namespace}")
                org_data = hf_client.get_org_details(namespace)
                if org_data:
                    author_kind = 'org'
                    auth_data = {'namespace': namespace, 'kind': 'org', 'raw_json': org_data}
                    db.upsert_author(auth_data)
                else:
                    user_data = hf_client.get_user_overview(namespace)
                    if user_data:
                        author_kind = 'user'
                        auth_data = {
                            'namespace': namespace,
                            'kind': 'user',
                            'num_followers': user_data.get('numFollowers'),
                            'is_pro': 1 if user_data.get('isPro') else 0,
                            'created_at': user_data.get('createdAt'),
                            'raw_json': user_data
                        }
                        db.upsert_author(auth_data)
                    else:
                        author_kind = 'unknown'
                        auth_data = {'namespace': namespace, 'kind': 'unknown', 'raw_json': {}}
                        db.upsert_author(auth_data)
            else:
                author_kind = auth_entry['kind']
                auth_data = auth_entry

            if author_kind == 'org':
                trust_tier = 3
            elif author_kind == 'user' and auth_data:
                followers = auth_data.get('num_followers') or 0
                is_pro = auth_data.get('is_pro') or 0
                if followers >= 200 or is_pro:
                    trust_tier = 2
                else:
                    trust_tier = 1

        logger.info(f"Author: {namespace} | Kind: {author_kind} | Tier: {trust_tier}")

        filter_trace = []
        tags = model_info.tags or []

        # --- Phase 0: Security & Hard Scope (Metadata) ---
        file_details = hf_client.get_model_file_details(model_id)
        if not filters.is_secure(file_details):
            logger.warning(f"Skipping {model_id}: Security check failed")
            continue

        if filters.is_generative_visual(model_info, tags):
            filter_trace.append("skip:generative_visual")

        if filters.is_excluded_content(model_id, tags): # Now checks strict NSFW
            filter_trace.append("skip:nsfw_excluded")

        if filters.is_export_or_conversion(model_id, tags, file_details):
            filter_trace.append("skip:export_conversion")

        # --- Phase 1: Deep Filters (Metadata + Files) ---
        params_est = filters.extract_parameter_count(model_info, file_details)
        if params_est and params_est > config.MAX_PARAMS_BILLIONS:
            filter_trace.append(f"skip:params_too_large_{params_est:.1f}B")

        if filter_trace:
            logger.info(f"Skipping {model_id}: {filter_trace}")
            continue

        # --- Phase 2: README Validation ---
        readme_content = hf_client.get_model_readme(model_id)
        if not readme_content:
            logger.info(f"Skipping {model_id}: No README")
            continue

        if filters.is_merge(model_id, readme_content):
            logger.info(f"Skipping {model_id}: Merge model")
            continue

        if filters.is_robotics_but_keep_vqa(model_info, tags, readme_content):
            logger.info(f"Skipping {model_id}: Robotics/Embodied")
            continue

        # --- Phase 3: Quality Gate (Tier-Dependent) ---
        yaml_meta = extract_yaml_front_matter(readme_content)
        links_present = filters.has_external_links(readme_content)
        info_score = filters.compute_info_score(readme_content, yaml_meta, tags, links_present)
        is_boilerplate = filters.is_boilerplate_readme(readme_content)
        is_roleplay = filters.is_roleplay(model_id, tags)

        final_status = 'processed'
        should_skip_quality = False

        if trust_tier <= 1: # Normal User
            if is_roleplay:
                should_skip_quality = True
                filter_trace.append("skip:roleplay_content")
            elif is_boilerplate:
                should_skip_quality = True
                filter_trace.append("skip:boilerplate_readme")
            elif info_score < 3 and not links_present:
                should_skip_quality = True
                filter_trace.append("skip:low_info_score")
        else: # Org / Strong User
            if is_roleplay:
                # Orgs doing RP? Allow analysis but maybe flag?
                # Assume if Org does it, it might be relevant (e.g. creative writing model)
                pass
            if is_boilerplate:
                final_status = 'review_required'

        if should_skip_quality:
            logger.info(f"Skipping {model_id}: Quality Gate Failed ({filter_trace[-1]})")
            continue

        # --- Phase 4: LLM Analysis ---
        logger.info(f"Analyzing {model_id} with LLM...")
        llm_result = llm_client.analyze_model(readme_content, tags, yaml_meta=yaml_meta, file_summary=file_details)

        if llm_result:
            evidence = llm_result.get('evidence', [])
            for item in evidence:
                quote = item.get('quote', '')
                if quote and quote not in readme_content:
                    logger.warning(f"Evidence fail: {quote[:30]}...")
                    final_status = 'review_required'
                    llm_result['confidence'] = 'low'
        else:
            logger.error(f"LLM analysis failed for {model_id}")
            final_status = 'error'

        if final_status == 'processed':
            logger.info(f"Analysis complete: Score {llm_result.get('specialist_score')}")

        model_data = {
            'id': model_id,
            'name': model_id.split('/')[-1],
            'author': namespace,
            'created_at': model_info.created_at,
            'last_modified': model_info.lastModified,
            'params_est': params_est,
            'hf_tags': tags,
            'llm_analysis': llm_result,
            'status': final_status,
            'namespace': namespace,
            'author_kind': author_kind,
            'trust_tier': trust_tier,
            'pipeline_tag': filters.get_pipeline_tag(model_info),
            'filter_trace': filter_trace,
            'report_notes': f"Evidence check: {'Passed' if final_status=='processed' else 'Failed'}" if llm_result else "LLM Failed"
        }

        if final_status != 'error':
            processed_models.append(model_data)
            if not args.dry_run:
                db.save_model(model_data)

    if processed_models:
        logger.info(f"Generating reports for {len(processed_models)} processed models...")
        md_path = reporter.generate_markdown_report(processed_models, date_str=date_str)
        reporter.export_csv(processed_models)

        if not args.dry_run or args.force_email:
            try:
                with open(md_path, 'r', encoding='utf-8') as f: md_content = f.read()
                mailer.send_report(md_content, date_str)
            except Exception as e: logger.error(f"Email failed: {e}")
    else:
        logger.info("No models processed.")

    if not args.dry_run:
        db.set_last_run_timestamp(current_run_time)

if __name__ == "__main__":
    main()
