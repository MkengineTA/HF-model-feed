# run_stats.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class ModelRef:
    model_id: str
    uploader: str

@dataclass
class SkipItem:
    model_id: str
    reason: str
    author: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WarnItem:
    model_id: str
    reason: str
    author: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunStats:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    candidates_total: int = 0
    queued: int = 0
    noop_unchanged: int = 0

    processed: int = 0
    skipped: int = 0

    # Warnings (do not stop processing)
    warned: int = 0
    warn_reasons: Counter[str] = field(default_factory=Counter)
    warn_items: List[WarnItem] = field(default_factory=list)

    # LLM metrics
    llm_analyzed: int = 0
    llm_succeeded: int = 0
    llm_failed: int = 0

    skip_reasons: Counter[str] = field(default_factory=Counter)
    skip_items: List[SkipItem] = field(default_factory=list)
    skip_reasons_by_uploader: Dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    processed_items: List[ModelRef] = field(default_factory=list)
    analyzed_items: List[ModelRef] = field(default_factory=list)

    # Tier 2 whitelist candidates (namespace -> metadata)
    tier2_candidates: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def record_candidate_batch(self, n: int) -> None:
        self.candidates_total += n

    def record_llm_analyzed(self, model_id: str, uploader: str) -> None:
        self.llm_analyzed += 1
        if len(self.analyzed_items) < 200:
            self.analyzed_items.append(ModelRef(model_id, uploader))

    def record_llm_succeeded(self) -> None:
        self.llm_succeeded += 1

    def record_llm_failed(self) -> None:
        self.llm_failed += 1

    def record_processed(self, model_id: str, uploader: str) -> None:
        self.processed += 1
        if len(self.processed_items) < 200:
            self.processed_items.append(ModelRef(model_id, uploader))

    def record_warn(self, model_id: str, reason: str, author: Optional[str] = None, **extra: Any) -> None:
        self.warned += 1
        self.warn_reasons[reason] += 1
        if len(self.warn_items) < 80:
            self.warn_items.append(WarnItem(model_id=model_id, reason=reason, author=author, extra=extra))

    def record_skip_reason_only(self, reason: str, author: Optional[str] = None, n: int = 1) -> None:
        if not reason:
            return
        self.skip_reasons[reason] += n
        if author:
            self.skip_reasons_by_uploader[author][reason] += n

    def record_skip(self, model_id: str, reason: str, author: Optional[str] = None, **extra: Any) -> None:
        self.skipped += 1
        self.skip_reasons[reason] += 1
        if author:
            self.skip_reasons_by_uploader[author][reason] += 1
        if len(self.skip_items) < 80:
            self.skip_items.append(SkipItem(model_id=model_id, reason=reason, author=author, extra=extra))

    def prolific_skipped_uploaders(self, reason: str, min_count: int) -> set[str]:
        results: set[str] = set()
        if not reason or min_count <= 0:
            return results

        for uploader, counter in self.skip_reasons_by_uploader.items():
            count = counter.get(reason, 0)
            if count < min_count:
                continue
            top_count = max(counter.values())
            # Only add if this reason is tied for the uploader's most common skip cause.
            if count == top_count:
                results.add(uploader)

        return results

    def top_skip_reasons(self, n: int = 12) -> List[Tuple[str, int]]:
        return self.skip_reasons.most_common(n)

    def top_warn_reasons(self, n: int = 12) -> List[Tuple[str, int]]:
        return self.warn_reasons.most_common(n)

    def record_tier2_candidate(
        self,
        namespace: str,
        followers: Optional[int] = None,
        is_pro: bool = False,
        model_id: Optional[str] = None,
    ) -> None:
        if namespace not in self.tier2_candidates:
            self.tier2_candidates[namespace] = {
                "followers": followers,
                "is_pro": is_pro,
                "model_ids": [],
                "count": 0,
            }
        
        if model_id and model_id not in self.tier2_candidates[namespace]["model_ids"]:
            self.tier2_candidates[namespace]["model_ids"].append(model_id)
        
        self.tier2_candidates[namespace]["count"] += 1

    def summary_line(self) -> str:
        dur_s = int((datetime.now(timezone.utc) - self.started_at).total_seconds())
        return (
            f"Run summary: discovered={self.candidates_total} | queued={self.queued} | noop={self.noop_unchanged} | "
            f"included={self.processed} | skipped={self.skipped} | warned={self.warned} | "
            f"llm_attempted={self.llm_analyzed} (ok={self.llm_succeeded}/fail={self.llm_failed}) | "
            f"duration={dur_s}s"
        )
