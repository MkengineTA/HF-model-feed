from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, DefaultDict

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
class RunStats:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # "Discovered" means unique candidates after dedupe across sources.
    candidates_total: int = 0
    # How many of the discovered candidates were queued for processing (new/updated/missing timestamp).
    queued: int = 0
    # Discovered but unchanged/already tracked -> no-op (not processed, not skipped).
    noop_unchanged: int = 0

    processed: int = 0
    skipped: int = 0
    # LLM metrics
    llm_analyzed: int = 0          # attempted
    llm_succeeded: int = 0
    llm_failed: int = 0

    skip_reasons: Counter[str] = field(default_factory=Counter)
    skip_items: List[SkipItem] = field(default_factory=list)
    # Aggregation for "Top skipped uploaders" with reasons
    skip_reasons_by_uploader: Dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    processed_items: List[ModelRef] = field(default_factory=list)
    analyzed_items: List[ModelRef] = field(default_factory=list)

    def record_candidate_batch(self, n: int) -> None:
        """
        Record discovered *unique* candidates (after dedupe across sources).
        Call exactly once per run with len(unique_candidates).
        """
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

    def record_skip_reason_only(self, reason: str, author: Optional[str] = None, n: int = 1) -> None:
        """
        Increment reason counters without incrementing skipped model count.
        Useful for "trace" reasons when you only want one skip per model,
        but still want multiple gate reasons to show up in aggregates.
        """
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
        # Limit stored skip items to prevent memory explosion/huge reports
        if len(self.skip_items) < 80:
            self.skip_items.append(SkipItem(model_id=model_id, reason=reason, author=author, extra=extra))

    def top_skip_reasons(self, n: int = 12) -> List[Tuple[str, int]]:
        return self.skip_reasons.most_common(n)

    def summary_line(self) -> str:
        dur_s = int((datetime.now(timezone.utc) - self.started_at).total_seconds())
        return (
            f"Run summary: discovered={self.candidates_total} | queued={self.queued} | noop={self.noop_unchanged} | "
            f"included={self.processed} | skipped={self.skipped} | "
            f"llm_attempted={self.llm_analyzed} (ok={self.llm_succeeded}/fail={self.llm_failed}) | "
            f"duration={dur_s}s"
        )
