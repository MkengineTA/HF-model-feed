from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import Counter
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
class RunStats:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    candidates_total: int = 0
    processed: int = 0
    skipped: int = 0
    llm_analyzed: int = 0

    skip_reasons: Counter[str] = field(default_factory=Counter)
    skip_items: List[SkipItem] = field(default_factory=list)

    processed_items: List[ModelRef] = field(default_factory=list)
    analyzed_items: List[ModelRef] = field(default_factory=list)

    def record_candidate_batch(self, n: int) -> None:
        self.candidates_total += n

    def record_llm_analyzed(self, model_id: str, uploader: str) -> None:
        self.llm_analyzed += 1
        if len(self.analyzed_items) < 200:
            self.analyzed_items.append(ModelRef(model_id, uploader))

    def record_processed(self, model_id: str, uploader: str) -> None:
        self.processed += 1
        if len(self.processed_items) < 200:
            self.processed_items.append(ModelRef(model_id, uploader))

    def record_skip(self, model_id: str, reason: str, author: Optional[str] = None, **extra: Any) -> None:
        self.skipped += 1
        self.skip_reasons[reason] += 1
        # Limit stored skip items to prevent memory explosion/huge reports
        if len(self.skip_items) < 80:
            self.skip_items.append(SkipItem(model_id=model_id, reason=reason, author=author, extra=extra))

    def top_skip_reasons(self, n: int = 12) -> List[Tuple[str, int]]:
        return self.skip_reasons.most_common(n)

    def summary_line(self) -> str:
        dur_s = int((datetime.now(timezone.utc) - self.started_at).total_seconds())
        return (
            f"Run summary: candidates={self.candidates_total} | "
            f"processed={self.processed} | skipped={self.skipped} | "
            f"llm_analyzed={self.llm_analyzed} | duration={dur_s}s"
        )
