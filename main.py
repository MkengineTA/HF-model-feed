# main.py
from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
import yaml
import dateutil.parser
import unicodedata
import string

import config
from utils import setup_logging
from database import Database
from hf_client import HFClient
from llm_client import LLMClient
from reporter import Reporter
from mailer import Mailer
import model_filters as filters
import namespace_policy
from run_stats import RunStats
from param_estimator import estimate_parameters, security_warnings

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


def _norm(s: str) -> str:
    # robust for evidence quote checks (incl. full-width punctuation)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = " ".join(s.split())
    return s


def quote_in_readme(quote: str, readme: str) -> bool:
    if not quote or not readme:
        return False

    # Fast exact
    if quote in readme:
        return True

    # Normalized containment
    q = _norm(quote)
    r = _norm(readme)
    if q in r:
        return True

    # Extra: strip punctuation for fuzzy matching
    trans = str.maketrans("", "", string.punctuation + "，。、：；「」『』（）()[]{}")
    q2 = q.translate(trans)
    r2 = r.translate(trans)
    return q2 in r2


def tree_has_readme(file_details) -> bool:
    if not isinstance(file_details, list):
        return False
    for item in file_details:
        path = (item.get("path") or "").strip().lower()
        if not path:
            continue
        if path.endswith("readme.md") or path.endswith("modelcard.md"):
            return True
        leaf = path.split("/")[-1]
        if leaf.startswith("readme"):
            return True
    return False


def apply_dynamic_blacklist(db: Database, stats: RunStats, dry_run: bool) -> None:
    if dry_run:
        return

    reason = "skip:no_readme"
    threshold = config.DYNAMIC_BLACKLIST_NO_README_MIN

    whitelist_snapshot = namespace_policy.get_whitelist()
    base_blacklist_snapshot = namespace_policy.get_base_blacklist()

    additions: dict[str, int] = {}
    for uploader, counter in stats.skip_reasons_by_uploader.items():
        count = counter.get(reason, 0)
        if count < threshold:
            continue
        top_count = max(counter.values()) if counter else 0
        if count != top_count:
            continue

        normalized = namespace_policy.normalize_namespace(uploader)
        if not normalized:
            continue
        if normalized in whitelist_snapshot or normalized in base_blacklist_snapshot:
            continue
        additions[normalized] = count

    if not additions:
        return

    existing_dynamic = db.get_dynamic_blacklist()
    new_additions = {ns: cnt for ns, cnt in additions.items() if ns not in existing_dynamic}
    db.upsert_dynamic_blacklist(additions, reason=reason)

    combined = existing_dynamic | set(additions.keys())
    namespace_policy.set_dynamic_blacklist(combined)
    if new_additions:
        logger.info(
            f"Dynamic blacklist updated with {len(new_additions)} uploader(s): {sorted(new_additions)}"
        )


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
    dynamic_blacklist = db.get_dynamic_blacklist()
    if dynamic_blacklist:
        namespace_policy.set_dynamic_blacklist(dynamic_blacklist)
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

    logger.info("Fetching models...")
    recent_models = hf_client.fetch_new_models(since=last_run_ts, limit=args.limit)
    updated_models = hf_client.fetch_recently_updated_models(since=last_run_ts, limit=args.limit)
    trending_ids = hf_client.fetch_trending_models(limit=10)
    paper_model_ids = hf_client.fetch_daily_papers(limit=20)

    candidates: dict[str, object] = {}

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
    processing_queue: list[tuple[str, object, str]] = []

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

    stats.queued = len(processing_queue)
    stats.noop_unchanged = max(0, len(candidates) - len(processing_queue))
    logger.info(f"Processing queue size: {len(processing_queue)}")

    processed_models: list[dict] = []
    author_run_cache: dict[str, tuple[str, object, int]] = {}
    seen_signatures: dict[str, str] = {}

    def skip(model_id: str, uploader: str | None, reason: str, **extra):
        stats.record_skip(model_id, reason, author=uploader, **extra)
        logger.info(f"Skipping {model_id}: {reason}")

    for model_id, model_info, reason in processing_queue:
        logger.info(f"Processing {model_id} ({reason})...")

        namespace = model_id.split("/")[0] if "/" in model_id else None
        uploader = namespace or "unknown"

        # ✅ Fix 2: niemals den ganzen Run durch einen Einzel-Fehler killen
        try:
            decision, ns_reason = namespace_policy.classify_namespace(uploader)
            is_whitelisted = (decision == "allow_whitelist")

            if decision == "deny_blacklist":
                skip(model_id, uploader, ns_reason or "skip:blacklisted_namespace")
                continue

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
                        elif org_data == {}:
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

                    if is_whitelisted:
                        trust_tier = 3

                    author_run_cache[namespace] = (author_kind, auth_data, trust_tier)

            logger.info(f"Author: {uploader} | Kind: {author_kind} | Tier: {trust_tier}")

            filter_trace: list[str] = []
            tags = getattr(model_info, "tags", None) or []

            # ---- Phase 0: hard scope / security warnings ----
            file_details = hf_client.get_model_file_details(model_id)

            # security warnings (never skip)
            for w_reason, w_extra in security_warnings(file_details):
                stats.record_warn(model_id, w_reason, author=uploader, **w_extra)
                filter_trace.append(w_reason)

            if filters.is_generative_visual(model_info, tags):
                filter_trace.append("skip:generative_visual")

            if filters.is_excluded_content(model_id, tags):
                filter_trace.append("skip:nsfw_excluded")

            if filters.is_export_or_conversion(model_id, tags, file_details):
                filter_trace.append("skip:export_conversion")

            if "skip:nsfw_excluded" in filter_trace:
                skip(model_id, uploader, "skip:nsfw_excluded", trace=filter_trace)
                continue

            if "skip:generative_visual" in filter_trace or "skip:export_conversion" in filter_trace:
                skip(model_id, uploader, filter_trace[0], trace=filter_trace)
                continue

            # ---- Phase 1: README ----
            readme_content = hf_client.get_model_readme(model_id)
            if not readme_content:
                has_tree = isinstance(file_details, list)
                has_readme_candidate = tree_has_readme(file_details)
                if has_readme_candidate or not has_tree:
                    stats.record_warn(model_id, "warn:readme_fetch_failed", author=uploader)
                    skip(model_id, uploader, "skip:readme_fetch_failed")
                else:
                    skip(model_id, uploader, "skip:no_readme")
                continue

            if filters.is_empty_or_stub_readme(readme_content):
                skip(model_id, uploader, "skip:empty_readme")
                continue

            if filters.is_merge(model_id, readme_content):
                skip(model_id, uploader, "skip:merge_model")
                continue

            if filters.is_robotics_but_keep_vqa(model_info, tags, readme_content):
                skip(model_id, uploader, "skip:robotics_embodied")
                continue

            if filters.is_blockassist_clone(model_id, namespace, tags, readme_content):
                skip(model_id, uploader, "skip:clone_of_canonical", canonical="gensyn/blockassist")
                continue

            # ---- Phase 2: Quality gate ----
            yaml_meta = extract_yaml_front_matter(readme_content)

            # ✅ Fix 1 (in main): yaml_meta niemals None lassen
            if not isinstance(yaml_meta, dict):
                yaml_meta = {}

            links_present = filters.has_external_links(readme_content)
            info_score = filters.compute_info_score(readme_content, yaml_meta, tags, links_present)
            is_boilerplate = filters.is_boilerplate_readme(readme_content)
            has_more_info = filters.has_more_info_needed(readme_content)
            is_roleplay = filters.is_roleplay(model_id, tags)

            if not is_whitelisted and trust_tier <= 1:
                if filters.is_unsloth_template_finetune(readme_content, tags=tags):
                    skip(model_id, uploader, "skip:template_finetune_unsloth")
                    continue
                if filters.is_finetune_from_quant_base(readme_content):
                    skip(model_id, uploader, "skip:finetune_from_quant_base")
                    continue

            if not is_whitelisted:
                if trust_tier <= 1:
                    if is_roleplay:
                        skip(model_id, uploader, "skip:roleplay_content")
                        continue
                    if is_boilerplate or has_more_info:
                        skip(model_id, uploader, "skip:boilerplate_readme")
                        continue
                    if info_score < 3 and not links_present:
                        skip(
                            model_id,
                            uploader,
                            "skip:low_info_score",
                            info_score=info_score,
                            links_present=links_present,
                        )
                        continue
                else:
                    if is_boilerplate or has_more_info:
                        skip(model_id, uploader, "skip:boilerplate_readme", trust_tier=trust_tier)
                        continue

            sig = filters.compute_repo_signature(readme_content, file_details)
            if sig in seen_signatures and seen_signatures[sig] != model_id:
                skip(model_id, uploader, "skip:duplicate_signature", canonical=seen_signatures[sig])
                continue
            seen_signatures[sig] = model_id

            # ---- Phase 2.5: Parameter gate (total + active) ----
            pe = estimate_parameters(hf_client.api, model_id, file_details)

            # Skip if thresholds exceeded (either one)
            if pe.total_b is not None and pe.total_b > config.MAX_TOTAL_PARAMS_BILLIONS:
                skip(
                    model_id,
                    uploader,
                    "skip:params_total_too_large",
                    total_b=pe.total_b,
                    max_total_b=config.MAX_TOTAL_PARAMS_BILLIONS,
                    source=pe.source,
                )
                continue
            if pe.active_b is not None and pe.active_b > config.MAX_ACTIVE_PARAMS_BILLIONS:
                skip(
                    model_id,
                    uploader,
                    "skip:params_active_too_large",
                    active_b=pe.active_b,
                    max_active_b=config.MAX_ACTIVE_PARAMS_BILLIONS,
                    source=pe.source,
                )
                continue

            # ---- Phase 3: LLM analysis ----
            stats.record_llm_analyzed(model_id, uploader)
            logger.info(f"Analyzing {model_id} with LLM...")
            llm_result = llm_client.analyze_model(
                readme_content,
                tags,
                yaml_meta=yaml_meta,
                file_summary=file_details,
            )

            if not llm_result:
                stats.record_llm_failed()
                skip(model_id, uploader, "skip:llm_error")
                continue
            stats.record_llm_succeeded()

            # Evidence check (internal)
            report_notes = []
            if config.LLM_REQUIRE_EVIDENCE:
                evidence = llm_result.get("evidence", []) or []
                any_mismatch = False
                if not evidence:
                    llm_result["confidence"] = "low"
                    report_notes.append("Evidence: Missing")
                else:
                    for item in evidence:
                        quote = item.get("quote", "")
                        if quote and not quote_in_readme(quote, readme_content):
                            any_mismatch = True
                            break
                    if any_mismatch:
                        llm_result["confidence"] = "low"
                        report_notes.append("Evidence: Mismatch")
                    else:
                        report_notes.append("Evidence: Passed")

            model_data = {
                "id": model_id,
                "name": model_id.split("/")[-1],
                "author": uploader,
                "created_at": getattr(model_info, "created_at", None),
                "last_modified": getattr(model_info, "lastModified", None),

                # keep legacy params_est as total
                "params_est": pe.total_b,
                "params_total_b": pe.total_b,
                "params_active_b": pe.active_b,
                "params_source": pe.source,

                "hf_tags": tags,
                "llm_analysis": llm_result,
                "status": "processed",
                "namespace": namespace,
                "author_kind": author_kind,
                "trust_tier": trust_tier,
                "pipeline_tag": filters.get_pipeline_tag(model_info),
                "filter_trace": filter_trace,
                "report_notes": " | ".join(report_notes) if report_notes else "",
            }

            if not args.dry_run:
                db.save_model(model_data)

            score = int((llm_result or {}).get("specialist_score", 0) or 0)
            keep_for_report = True
            if config.EXCLUDE_REVIEW_REQUIRED and model_data.get("status") != "processed":
                keep_for_report = False
            if score < config.MIN_SPECIALIST_SCORE:
                keep_for_report = False

            if keep_for_report:
                processed_models.append(model_data)
                stats.record_processed(model_id, uploader)
            else:
                skip(
                    model_id,
                    uploader,
                    "skip:report_filtered",
                    score=score,
                    min_score=config.MIN_SPECIALIST_SCORE,
                )

        except Exception as e:
            logger.exception(f"Unhandled error while processing {model_id}: {e}")
            skip(model_id, uploader, "skip:exception", error=str(e))
            continue

    apply_dynamic_blacklist(db, stats, args.dry_run)

    if processed_models or stats.skipped > 0 or stats.warned > 0:
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
