import logging
import argparse
from datetime import datetime, timezone
import re
import yaml
import dateutil.parser

import config
from utils import setup_logging
from database import Database
from hf_client import HFClient
from llm_client import LLMClient
from reporter import Reporter
from mailer import Mailer
import model_filters as filters
from namespace_policy import classify_namespace
from run_stats import RunStats

logger = setup_logging()


def extract_yaml_front_matter(readme_text: str | None):
    if not readme_text:
        return None
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", readme_text, re.DOTALL)
    if match:
        try:
            return yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
    return None


def quote_in_readme(quote: str, readme: str) -> bool:
    if not quote or not readme:
        return False
    if quote in readme:
        return True
    q = " ".join(quote.split()).lower()
    r = " ".join(readme.split()).lower()
    return q in r


def main():
    parser = argparse.ArgumentParser(description="Edge AI Scout & Specialist Model Monitor")
    parser.add_argument("--limit", type=int, default=1000, help="Safety limit for API fetching (default: 1000)")
    parser.add_argument("--dry-run", action="store_true", help="Do not save to DB")
    parser.add_argument("--force-email", action="store_true", help="Send email even in dry-run mode")
    args = parser.parse_args()

    stats = RunStats()
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
        enable_reasoning=config.LLM_ENABLE_REASONING,
    )

    reporter = Reporter()
    mailer = Mailer()

    last_run_ts = db.get_last_run_timestamp()
    logger.info(f"Last successful run: {last_run_ts}")

    # ---- Fetching candidates ----
    logger.info("Fetching models...")
    recent_models = hf_client.fetch_new_models(since=last_run_ts, limit=args.limit)
    updated_models = hf_client.fetch_recently_updated_models(since=last_run_ts, limit=args.limit)
    trending_ids = hf_client.fetch_trending_models(limit=10)
    paper_model_ids = hf_client.fetch_daily_papers(limit=20)

    candidates = {}

    def add_candidate(info):
        if info and getattr(info, "id", None) and info.id not in candidates:
            candidates[info.id] = info

    for m in recent_models:
        add_candidate(m)
    for m in updated_models:
        add_candidate(m)

    extra_ids = set(trending_ids + paper_model_ids)
    for mid in extra_ids:
        if mid not in candidates:
            info = hf_client.get_model_info(mid)
            add_candidate(info)

    stats.record_candidate_batch(len(candidates))
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
            api_last_mod_dt = getattr(model_info, "lastModified", None)
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
    author_run_cache = {}

    def skip(model_id: str, uploader: str | None, reason: str, **extra):
        stats.record_skip(model_id, reason, author=uploader, **extra)
        logger.info(f"Skipping {model_id}: {reason}")

    stats.record_candidate_batch(len(processing_queue)) # Wait, adding queue size to total?
    # candidates_total is sum of fetch batches.
    # Here we are logging candidates passed to process queue (subset of total candidates that need update/new).
    # The previous `record_candidate_batch` logged total found.
    # Let's just log queue size.
    logger.info(f"Processing queue size: {len(processing_queue)}")

    for model_id, model_info, reason in processing_queue:
        logger.info(f"Processing {model_id} ({reason})...")

        namespace = model_id.split("/")[0] if "/" in model_id else None
        uploader = namespace or "unknown"

        def record_skip_local(primary_reason: str, *, trace=None, **extra):
            if trace and isinstance(trace, list) and trace:
                stats.record_skip(model_id, trace[0], author=uploader, trace=trace, **extra)
                for r in trace[1:]:
                    stats.skip_reasons[r] += 1
            else:
                stats.record_skip(model_id, primary_reason, author=uploader, **extra)

        # ---- Namespace policy (early) ----
        decision, ns_reason = classify_namespace(uploader)
        is_whitelisted = (decision == "allow_whitelist")

        if decision == "deny_blacklist":
            logger.info(f"Skipping {model_id}: {ns_reason}")
            record_skip_local(ns_reason or "skip:blacklisted_namespace")
            continue

        # Optional legacy exclusion
        if namespace and namespace.lower() in {x.lower() for x in getattr(config, "EXCLUDED_NAMESPACES", set())}:
            skip(model_id, namespace, "skip:excluded_namespace")
            continue

        # ---- Author tier cache ----
        trust_tier = 1
        author_kind = "unknown"

        if namespace:
            if namespace in author_run_cache:
                author_kind, auth_data, trust_tier = author_run_cache[namespace]
            else:
                auth_entry = db.get_author(namespace)
                auth_data = None
                cache_valid = False

                if auth_entry:
                    try:
                        last_checked = dateutil.parser.parse(str(auth_entry["last_checked"]))
                        age_days = (datetime.now(timezone.utc) - last_checked).days
                        kind_db = auth_entry["kind"]

                        if kind_db in ("user", "org") and age_days < 14:
                            cache_valid = True
                        if kind_db == "unknown":
                            cache_valid = False
                    except Exception:
                        cache_valid = False

                if not cache_valid:
                    logger.info(f"Validating author: {namespace}")
                    org_data = hf_client.get_org_details(namespace)

                    if org_data:
                        author_kind = "org"
                        auth_data = {"namespace": namespace, "kind": "org", "raw_json": org_data}
                        db.upsert_author(auth_data)

                    elif org_data == {}:  # Not an org -> user check
                        user_data = hf_client.get_user_overview(namespace)

                        if user_data:
                            author_kind = "user"
                            auth_data = {
                                "namespace": namespace,
                                "kind": "user",
                                "num_followers": user_data.get("numFollowers"),
                                "is_pro": 1 if user_data.get("isPro") else 0,
                                "created_at": user_data.get("createdAt"),
                                "raw_json": user_data,
                            }
                            db.upsert_author(auth_data)

                        elif user_data == {}:
                            author_kind = "unknown"
                            auth_data = {"namespace": namespace, "kind": "unknown", "raw_json": {}}
                            db.upsert_author(auth_data)

                        else:
                            author_kind = "unknown"
                            auth_data = None
                    else:
                        author_kind = "unknown"
                        auth_data = None
                else:
                    author_kind = auth_entry["kind"]
                    auth_data = dict(auth_entry)

                if author_kind == "org":
                    trust_tier = 3
                elif author_kind == "user" and auth_data:
                    followers = auth_data.get("num_followers") or 0
                    is_pro = auth_data.get("is_pro") or 0
                    trust_tier = 2 if (followers >= 200 or is_pro) else 1
                else:
                    trust_tier = 1

                author_run_cache[namespace] = (author_kind, auth_data, trust_tier)

        logger.info(f"Author: {uploader} | Kind: {author_kind} | Tier: {trust_tier}")

        filter_trace = []
        tags = getattr(model_info, "tags", None) or []

        # ---- Phase 0: Security & hard scope ----
        file_details = hf_client.get_model_file_details(model_id)
        if not filters.is_secure(file_details):
            logger.warning(f"Skipping {model_id}: Security check failed")
            skip(model_id, namespace, "skip:security_failed")
            continue

        if filters.is_generative_visual(model_info, tags):
            filter_trace.append("skip:generative_visual")

        if filters.is_excluded_content(model_id, tags):
            filter_trace.append("skip:nsfw_excluded")

        if filters.is_export_or_conversion(model_id, tags, file_details):
            filter_trace.append("skip:export_conversion")

        params_est = filters.extract_parameter_count(model_info, file_details)
        if params_est and params_est > config.MAX_PARAMS_BILLIONS:
            filter_trace.append(f"skip:params_too_large_{params_est:.1f}B")

        if "skip:nsfw_excluded" in filter_trace:
            logger.info(f"Skipping {model_id}: {filter_trace}")
            record_skip_local("skip:nsfw_excluded", trace=filter_trace)
            continue

        if any(x.startswith("skip:params_too_large_") for x in filter_trace):
            logger.info(f"Skipping {model_id}: {filter_trace}")
            record_skip_local(filter_trace[0], trace=filter_trace)
            continue

        if not is_whitelisted:
            if "skip:generative_visual" in filter_trace or "skip:export_conversion" in filter_trace:
                logger.info(f"Skipping {model_id}: {filter_trace}")
                record_skip_local(filter_trace[0], trace=filter_trace)
                continue

        # ---- Phase 1: README ----
        readme_content = hf_client.get_model_readme(model_id)
        if not readme_content:
            logger.info(f"Skipping {model_id}: No README")
            skip(model_id, namespace, "skip:no_readme")
            continue

        if filters.is_empty_or_stub_readme(readme_content):
            logger.info(f"Skipping {model_id}: Empty/Stub README")
            skip(model_id, namespace, "skip:empty_readme")
            continue

        if filters.is_merge(model_id, readme_content):
            skip(model_id, namespace, "skip:merge_model")
            continue

        if filters.is_robotics_but_keep_vqa(model_info, tags, readme_content):
            skip(model_id, namespace, "skip:robotics_embodied")
            continue

        # ---- Phase 2: Quality gate ----
        yaml_meta = extract_yaml_front_matter(readme_content)
        links_present = filters.has_external_links(readme_content)
        info_score = filters.compute_info_score(readme_content, yaml_meta, tags, links_present)
        is_boilerplate = filters.is_boilerplate_readme(readme_content)
        has_more_info = filters.has_more_info_needed(readme_content)
        is_roleplay = filters.is_roleplay(model_id, tags)

        if not is_whitelisted:
            if trust_tier <= 1:
                if is_roleplay:
                    skip(model_id, namespace, "skip:roleplay_content")
                    continue
                if is_boilerplate or has_more_info:
                    skip(model_id, namespace, "skip:boilerplate_readme")
                    continue
                if info_score < 3 and not links_present:
                    skip(model_id, namespace, "skip:low_info_score", info_score=info_score, links_present=links_present)
                    continue
            else:
                if is_boilerplate or has_more_info:
                    skip(model_id, namespace, "skip:boilerplate_readme", trust_tier=trust_tier)
                    continue

        # ---- Phase 3: LLM analysis ----
        stats.record_llm_analyzed(model_id, uploader)
        logger.info(f"Analyzing {model_id} with LLM...")
        llm_result = llm_client.analyze_model(
            readme_content, tags, yaml_meta=yaml_meta, file_summary=file_details
        )

        if not llm_result:
            logger.error(f"LLM analysis failed for {model_id}")
            skip(model_id, namespace, "skip:llm_error")
            continue

        report_note_evidence = "Evidence check: Passed"
        evidence = llm_result.get("evidence", [])
        any_mismatch = False
        for item in evidence:
            quote = item.get("quote", "")
            if quote and not quote_in_readme(quote, readme_content):
                logger.warning(f"Evidence mismatch: {quote[:60]}...")
                any_mismatch = True

        if any_mismatch:
            llm_result["confidence"] = "low"
            report_note_evidence = "Evidence check: Mismatch"

        logger.info(f"Analysis complete: Score {llm_result.get('specialist_score')}")

        model_data = {
            "id": model_id,
            "name": model_id.split("/")[-1],
            "author": uploader,
            "created_at": getattr(model_info, "created_at", None),
            "last_modified": getattr(model_info, "lastModified", None),
            "params_est": params_est,
            "hf_tags": tags,
            "llm_analysis": llm_result,
            "status": "processed",
            "namespace": namespace,
            "author_kind": author_kind,
            "trust_tier": trust_tier,
            "pipeline_tag": filters.get_pipeline_tag(model_info),
            "filter_trace": filter_trace,
            "report_notes": report_note_evidence,
        }

        if not args.dry_run:
            db.save_model(model_data)

        min_score = getattr(config, "MIN_SPECIALIST_SCORE", 0)
        exclude_review = getattr(config, "EXCLUDE_REVIEW_REQUIRED", False)

        score = int((llm_result or {}).get("specialist_score", 0) or 0)
        keep_for_report = True

        if exclude_review and model_data.get("status") != "processed":
            keep_for_report = False
        if score < min_score:
            keep_for_report = False

        if keep_for_report:
            processed_models.append(model_data)
            stats.record_processed(model_id, uploader)
        else:
            skip(model_id, namespace, "skip:report_filtered", score=score, min_score=min_score)

    if processed_models or stats.skipped > 0: # Always report if we did anything
        logger.info(f"Generating reports for {len(processed_models)} processed models...")

        md_path = reporter.generate_full_report(stats, processed_models, date_str=date_str)
        reporter.export_csv(processed_models)

        logger.info(stats.summary_line())

        if not args.dry_run or args.force_email:
            try:
                md_content = md_path.read_text(encoding="utf-8")
                mailer.send_report(md_content, date_str)
            except Exception as e:
                logger.error(f"Email failed: {e}")
    else:
        logger.info("No candidates processed.")
        logger.info(stats.summary_line())

    if not args.dry_run:
        db.set_last_run_timestamp(current_run_time)


if __name__ == "__main__":
    main()
