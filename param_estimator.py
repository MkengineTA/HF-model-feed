# param_estimator.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

import config

logger = logging.getLogger("EdgeAIScout")

_EXECUTABLE_EXTS = (".exe", ".msi", ".bat", ".cmd", ".dll")
_SCRIPT_EXTS = (".sh", ".py", ".js", ".ps1", ".php", ".pl")

_WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth")
_INDEX_FILES = ("model.safetensors.index.json",)

@dataclass
class ParamEstimate:
    total_params: Optional[int]
    active_params: Optional[int]
    total_b: Optional[float]
    active_b: Optional[float]
    source: str  # "safetensors_metadata" | "config_heuristic" | "filesize_fallback" | "unknown"
    dtype_breakdown: Optional[dict[str, int]]
    is_moe: bool
    experts: str
    notes: list[str]

def _safe_float_b(x: Optional[int]) -> Optional[float]:
    if x is None:
        return None
    return round(x / 1e9, 3)

def _load_json_if_exists(repo_id: str, filename: str) -> Optional[dict[str, Any]]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _detect_moe(cfg: dict[str, Any]) -> tuple[bool, int, int, int]:
    """
    Returns: (is_moe, num_experts, experts_per_tok, shared_experts)
    """
    txt = json.dumps(cfg).lower()

    moe_node = cfg.get("moe", {}) if isinstance(cfg.get("moe", {}), dict) else {}
    num_experts = (
        cfg.get("num_local_experts")
        or cfg.get("num_experts")
        or moe_node.get("num_experts")
        or 1
    )
    experts_per_tok = (
        cfg.get("num_experts_per_tok")
        or cfg.get("num_experts_per_token")
        or moe_node.get("num_experts_per_tok")
        or 1
    )
    shared_experts = (
        cfg.get("num_shared_experts")
        or moe_node.get("num_shared_experts")
        or 0
    )

    is_moe = bool(num_experts and int(num_experts) > 1) or ("mixture-of-experts" in txt) or ("moe" in txt)
    return is_moe, int(num_experts or 1), int(experts_per_tok or 1), int(shared_experts or 0)

def _heuristic_params_from_config(cfg: dict[str, Any]) -> tuple[Optional[int], Optional[int], bool, str, list[str]]:
    """
    Returns (total_params, active_params, is_moe, experts_str, notes)

    Heuristic: optimized for decoder-only LMs + common MoE configs.
    If we cannot extract enough, returns (None, None, ...)
    """
    notes: list[str] = []

    # Try common LM dims
    h = cfg.get("hidden_size") or cfg.get("dim") or cfg.get("d_model")
    if not h:
        return None, None, False, "Unknown", ["config_missing:hidden_size/dim/d_model"]

    # Layers (decoder-only)
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layers")
    if not layers:
        # Sometimes `layer_types` exists
        lt = cfg.get("layer_types")
        if isinstance(lt, list) and len(lt) > 0:
            layers = len(lt)
    if not layers:
        return None, None, False, "Unknown", ["config_missing:num_hidden_layers/n_layers"]

    vocab = cfg.get("vocab_size") or 32000

    # Intermediate size (FFN)
    moe_node = cfg.get("moe", {}) if isinstance(cfg.get("moe", {}), dict) else {}
    inter = (
        cfg.get("intermediate_size")
        or cfg.get("hidden_dim")
        or cfg.get("decoder_ffn_dim")
        or moe_node.get("expert_hidden_dim")
    )
    if not inter:
        return None, None, False, "Unknown", ["config_missing:intermediate_size"]

    is_moe, num_experts, experts_per_tok, shared_experts = _detect_moe(cfg)
    experts_str = f"{experts_per_tok}/{num_experts}" if is_moe else "Dense"

    # Detect how many MoE layers vs dense layers
    moe_layers_list = cfg.get("is_moe_layer")
    if isinstance(moe_layers_list, list) and len(moe_layers_list) > 0:
        moe_count = sum(1 for x in moe_layers_list if x)
        dense_count = len(moe_layers_list) - moe_count
    else:
        dense_count = cfg.get("first_k_dense_replace") or moe_node.get("first_k_dense_replace") or 0
        try:
            dense_count = int(dense_count)
        except Exception:
            dense_count = 0
        moe_count = int(layers) - int(dense_count)

    # Embedding
    embedding_p = int(vocab) * int(h)

    # Attention params (very rough): Wq/Wk/Wv/Wo ~ 4*h*h
    # (we keep your spirit, but keep it explicit "approx")
    base_layer_p = 4 * (int(h) ** 2)

    def mlp_size(hidden: int, intermediate: int) -> int:
        # SwiGLU: up, gate, down => 3 * h * inter
        return 3 * hidden * intermediate

    dense_mlp = mlp_size(int(h), int(inter))
    expert_p = dense_mlp  # per expert FFN block

    # Active per-token MLP for MoE layers
    active_mlp = (int(experts_per_tok) + int(shared_experts)) * expert_p
    total_mlp = (int(num_experts) + int(shared_experts)) * expert_p

    # Totals
    total_active = embedding_p + (int(layers) * base_layer_p) + (int(dense_count) * dense_mlp) + (int(moe_count) * active_mlp)
    total_params = embedding_p + (int(layers) * base_layer_p) + (int(dense_count) * dense_mlp) + (int(moe_count) * total_mlp)

    notes.append("heuristic:decoder_only_transformer_or_moe")
    if is_moe:
        notes.append(f"moe_layers={moe_count},dense_layers={dense_count}")

    return int(total_params), int(total_active), is_moe, experts_str, notes

def _estimate_from_filesize(file_details: list[dict[str, Any]] | None) -> Optional[int]:
    if not isinstance(file_details, list) or not file_details:
        return None
    total_size = 0
    for f in file_details:
        p = (f.get("path") or "").lower()
        if p.endswith(_WEIGHT_EXTS):
            total_size += int(f.get("size") or 0)
    if total_size <= 0:
        return None
    # params ~= bytes / bytes_per_param
    return int(total_size / max(config.BYTES_PER_PARAM_FALLBACK, 0.1))

def estimate_parameters(api, repo_id: str, file_details: list[dict[str, Any]] | None) -> ParamEstimate:
    notes: list[str] = []
    total_params: Optional[int] = None
    active_params: Optional[int] = None
    source = "unknown"
    dtype_breakdown: Optional[dict[str, int]] = None

    # 1) Exact total params from safetensors metadata (when available)
    if config.PARAMS_PREFER_SAFETENSORS_META:
        try:
            meta = api.get_safetensors_metadata(repo_id)
            # SafetensorsRepoMetadata.parameter_count: dict[str,int] per dtype
            dtype_breakdown = dict(meta.parameter_count or {})
            total_params = int(sum(dtype_breakdown.values())) if dtype_breakdown else None
            if total_params:
                source = "safetensors_metadata"
                notes.append("total:exact_from_safetensors_metadata")
        except HfHubHTTPError as e:
            # Not a safetensors-root repo, or gated, etc.
            notes.append(f"safetensors_metadata_unavailable:{getattr(e, 'response', None) and e.response.status_code}")
        except Exception as e:
            notes.append(f"safetensors_metadata_error:{type(e).__name__}")

    # 2) Active params via config heuristic (MoE), and possibly total fallback if safetensors missing
    cfg = _load_json_if_exists(repo_id, "config.json") or _load_json_if_exists(repo_id, "params.json")
    is_moe = False
    experts_str = "Unknown"
    if cfg:
        t, a, is_moe, experts_str, n2 = _heuristic_params_from_config(cfg)
        notes.extend(n2)
        if a:
            active_params = a
        if (not total_params) and t:
            total_params = t
            source = "config_heuristic"
            notes.append("total:heuristic_from_config")

    # 3) If still no total, use file size fallback
    if not total_params:
        fs_est = _estimate_from_filesize(file_details)
        if fs_est:
            total_params = fs_est
            source = "filesize_fallback"
            notes.append("total:filesize_fallback")

    # 4) If dense or unknown, set active=total when missing
    if not active_params and total_params:
        active_params = total_params
        if is_moe:
            notes.append("active_fallback_to_total_despite_moe")
        else:
            notes.append("active=total_dense_or_unknown")

    return ParamEstimate(
        total_params=total_params,
        active_params=active_params,
        total_b=_safe_float_b(total_params),
        active_b=_safe_float_b(active_params),
        source=source,
        dtype_breakdown=dtype_breakdown,
        is_moe=is_moe,
        experts=experts_str,
        notes=notes,
    )

def security_warnings(file_details: list[dict[str, Any]] | None) -> list[tuple[str, dict[str, Any]]]:
    """
    Convert security situation to WARNINGS (never skip).
    """
    warnings: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(file_details, list) or not file_details:
        return warnings

    for f in file_details:
        path = (f.get("path") or "")
        low = path.lower()

        # HF security scan flags (if present)
        sec = f.get("securityFileStatus") or f.get("security")  # API variants
        status = None
        if isinstance(sec, dict):
            status = sec.get("status") or sec.get("result")
        elif isinstance(sec, str):
            status = sec

        if config.SECURITY_WARN_ON_HF_SCAN_FLAGS and status in {"unsafe", "malicious", "infected"}:
            warnings.append(("warn:security_scan_flagged", {"file": path, "status": status}))

        if config.SECURITY_WARN_ON_EXECUTABLES and low.endswith(_EXECUTABLE_EXTS):
            warnings.append(("warn:executable_file_present", {"file": path}))

        if config.SECURITY_WARN_ON_SCRIPTS and low.endswith(_SCRIPT_EXTS):
            warnings.append(("warn:script_file_present", {"file": path}))

    return warnings
