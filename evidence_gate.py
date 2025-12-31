from __future__ import annotations

from typing import Any, Dict, List, Tuple

def evidence_gate(llm_analysis: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Returns (ok, failures).
    Expectation: LLM returns structured output with at least:
      - evidence: list[ { claim, quote } ]
    """
    failures: List[str] = []

    # Check for evidence field
    ev = llm_analysis.get("evidence")
    if not isinstance(ev, list) or len(ev) == 0:
        failures.append("missing:evidence_list")
    else:
        for i, e in enumerate(ev):
            if not isinstance(e, dict):
                failures.append(f"evidence[{i}]:not_a_dict")
                continue
            if not e.get("quote"):
                failures.append(f"evidence[{i}]:missing_quote")
            # Claim is optional but good practice? User said "claim, quote".
            # My Prompt requests { "claim": ..., "quote": ... }
            if not e.get("claim"):
                failures.append(f"evidence[{i}]:missing_claim")

    return (len(failures) == 0, failures)
