# digest.py
"""
Multi-recipient digest scheduling and dispatch.

This module handles:
- Determining which subscribers should receive an email on a given day
- Grouping subscribers by (window_hours, language, report_type) to minimize report generation
- Dispatching grouped digests from DB-only queries (no additional HF/LLM API calls)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

import config
from config import NewsletterSubscriber, get_newsletter_subscribers
from database import Database
from reporter import Reporter
from mailer import Mailer
from run_stats import RunStats

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("EdgeAIScout")

# Day name mapping (lowercase)
DAY_NAME_MAP = {
    0: "mon",
    1: "tue",
    2: "wed",
    3: "thu",
    4: "fri",
    5: "sat",
    6: "sun",
}


def get_current_day_name(tz_name: str = "Europe/Berlin") -> str:
    """Get lowercase day name (mon/tue/...) in the specified timezone."""
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning(f"Invalid timezone '{tz_name}', falling back to UTC")
        tz = timezone.utc
    
    now = datetime.now(tz)
    return DAY_NAME_MAP[now.weekday()]


def get_subscribers_for_today(
    subscribers: List[NewsletterSubscriber],
    tz_name: str = "Europe/Berlin",
) -> List[Tuple[NewsletterSubscriber, int]]:
    """
    Filter subscribers who should receive an email today and compute their window_hours.
    
    Returns:
        List of (subscriber, window_hours) tuples for subscribers scheduled to receive today
    """
    day = get_current_day_name(tz_name)
    result = []
    
    for sub in subscribers:
        if day in sub.send_days:
            window = sub.get_window_hours_for_day(day)
            result.append((sub, window))
    
    return result


def group_subscribers(
    subscribers_with_windows: List[Tuple[NewsletterSubscriber, int]],
) -> Dict[Tuple[int, str, str], List[str]]:
    """
    Group subscribers by (window_hours, language, report_type) to minimize report generation.
    
    Returns:
        Dict mapping (window_hours, language, type) -> list of email addresses
    """
    groups: Dict[Tuple[int, str, str], List[str]] = defaultdict(list)
    
    for sub, window in subscribers_with_windows:
        key = (window, sub.language, sub.type)
        groups[key].append(sub.email)
    
    return dict(groups)


def dispatch_digests(
    db: Database,
    stats: RunStats,
    reporter: Reporter,
    mailer: Mailer,
    date_str: str,
    processed_models_current_run: Optional[List[Dict[str, Any]]] = None,
    force_send: bool = False,
) -> int:
    """
    Dispatch digests to all subscribers scheduled for today.
    
    For each unique (window_hours, language, type) group:
    1. If window_hours == 24 and we have processed_models_current_run, use those (no extra DB query)
    2. Otherwise, query DB for models by processed_at window
    3. Generate report with appropriate language and type
    4. Send to all emails in the group
    
    Args:
        db: Database instance
        stats: Current run stats (for debug reports)
        reporter: Reporter instance
        mailer: Mailer instance
        date_str: Date string for reports
        processed_models_current_run: Models processed in current run (for 24h window optimization)
        force_send: If True, send even if no models (for testing)
    
    Returns:
        Number of emails sent
    """
    subscribers = get_newsletter_subscribers()
    if not subscribers:
        logger.info("No newsletter subscribers configured.")
        return 0
    
    tz_name = config.NEWSLETTER_TIMEZONE
    today_subscribers = get_subscribers_for_today(subscribers, tz_name)
    
    if not today_subscribers:
        logger.info(f"No subscribers scheduled to receive email today ({get_current_day_name(tz_name)}).")
        return 0
    
    logger.info(f"Found {len(today_subscribers)} subscriber(s) scheduled for today.")
    
    # Group subscribers
    groups = group_subscribers(today_subscribers)
    logger.info(f"Grouped into {len(groups)} unique report configuration(s).")
    
    emails_sent = 0
    
    for (window_hours, language, report_type), emails in groups.items():
        logger.info(
            f"Generating report: window={window_hours}h, language={language}, "
            f"type={report_type}, recipients={len(emails)}"
        )
        
        # Get models for this window
        if window_hours == 24 and processed_models_current_run is not None:
            # Use current run's models directly (no extra DB query)
            models = processed_models_current_run
            logger.info(f"Using {len(models)} models from current run.")
        else:
            # Query DB for models in window
            models = db.get_models_by_processed_window(
                window_hours=window_hours,
                min_specialist_score=config.MIN_SPECIALIST_SCORE,
                exclude_review_required=config.EXCLUDE_REVIEW_REQUIRED,
            )
            logger.info(f"Queried {len(models)} models from DB for {window_hours}h window.")
        
        if not models and not force_send:
            logger.info(f"No models for this group, skipping email.")
            continue
        
        # Generate report
        try:
            report_path = reporter.generate_full_report(
                stats=stats,
                processed_models=models,
                date_str=date_str,
                language=language,
                report_type=report_type,
            )
            
            md_content = report_path.read_text(encoding="utf-8")
            
            # Send to all recipients in this group
            mailer.send_report(
                markdown_content=md_content,
                date_str=date_str,
                recipients=emails,
                language=language,
            )
            
            emails_sent += len(emails)
            
        except Exception as e:
            logger.error(f"Failed to generate/send report for group: {e}")
    
    logger.info(f"Digest dispatch complete: {emails_sent} email(s) sent.")
    return emails_sent
