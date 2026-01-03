# model_filters.py
from __future__ import annotations

import re
import hashlib
from typing import Optional, Any
from urllib.parse import urlparse

# Robotics-specific terms for substring matching (safe for use with `in`)
# Only long, unambiguous terms that won't match common words
# Removed generic terms: "kinematics", "actuator", "manipulation" (can appear in non-robotics contexts)
ROBOTICS_KEYWORDS_SUBSTRING = [
    "robot", "robotics", "robotik", "roboter",
    "reinforcement learning", "reinforcement-learning", "sim2real", "lidar", "slam",
    "openvla", "lerobot", "panda-reach", "pandareach",
    "gym-robotics", "pybullet", "mujoco", "isaacgym"
]

# Short tokens that require word boundary matching to avoid false positives
# "rl" matches "world", "real-world"; "ros" matches "across", "micros"
# Each tuple is (pattern, human-readable label for logs)
ROBOTICS_KEYWORDS_REGEX = [
    (re.compile(r"\brl\b", re.IGNORECASE), "rl"),        # standalone "rl" only
    (re.compile(r"\bros2\b", re.IGNORECASE), "ros2"),    # "ros2" as standalone word
    (re.compile(r"\bros\b", re.IGNORECASE), "ros"),      # "ros" as standalone word
]

# Pipelines that indicate robotics/embodied AI
ROBOTICS_PIPELINES = [
    "reinforcement-learning",
]

VISUAL_KEYWORDS = [
    "diffusion", "stable diffusion", "flux", "gan", "text-to-image",
    "image-to-image", "text-to-video", "inpainting", "super resolution",
    "controlnet", "lora", "unet", "vae", "diffusers"
]

VISUAL_PIPELINES = [
    "text-to-image", "image-to-text", "image-to-image",
    "text-to-video", "video-to-text", "unconditional-image-generation",
    "diffusers", "controlnet"
]

QUANT_NAME_PATTERNS = [
    re.compile(r"(^|[-_])(GGUF|GGML|AWQ|GPTQ|EXL2|ONNX)($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])(Q\d_[K0-9A-Z]+|Q\d)($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])(int4|int8|fp16|bf16)($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])(\d+bit)($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])(hqq|quip|squeeze)($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])(I|T)Q\d(_[A-Z0-9]+)*($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])Q[2-8](_K(_(XXS|XS|S|M|L|XL))?|_[01])($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])BF16($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])FP8($|[-_])", re.IGNORECASE),
    re.compile(r"(^|[-_])FP16($|[-_])", re.IGNORECASE),
]

UNSLOTH_TEMPLATE_MARKERS = [
    "this llama model was trained 2x faster with unsloth",
    "trained 2x faster with unsloth",
    "uploaded model",
    "finetuned from model",
]
FINETUNED_FROM_RE = re.compile(r"finetuned from model\s*:?\s*(.+)", re.IGNORECASE)
WALLET_ADDR_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")

def get_pipeline_tag(model_info) -> str:
    return getattr(model_info, "pipeline_tag", "") or ""

def is_generative_visual(model_info, tags) -> bool:
    pipeline = get_pipeline_tag(model_info)
    if pipeline in VISUAL_PIPELINES:
        return True
    tagset = {t.lower() for t in (tags or [])}
    if any(k in tagset for k in VISUAL_KEYWORDS):
        return True
    mid = (getattr(model_info, "modelId", "") or "").lower()
    if "diffusion" in mid or "flux" in mid or "lora" in mid:
        return True
    return False

def is_excluded_content(model_id: str, tags) -> bool:
    tagset = {t.lower() for t in (tags or [])}
    if "nsfw" in tagset:
        return True
    return False

def _check_robotics_keywords(text: str) -> str | None:
    """Check if text contains robotics keywords.
    
    Returns the matched keyword/pattern if found, None otherwise.
    Uses substring matching for long terms and regex with word boundaries for short tokens.
    """
    text_lower = text.lower()
    # Check substring keywords (safe, long terms)
    for k in ROBOTICS_KEYWORDS_SUBSTRING:
        if k in text_lower:
            return k
    # Check regex patterns (short tokens with word boundaries)
    for pattern, label in ROBOTICS_KEYWORDS_REGEX:
        if pattern.search(text_lower):
            return label  # Return human-readable label instead of raw pattern
    return None

def _check_robotics_tag(tag: str) -> str | None:
    """Check if a single tag matches robotics keywords.
    
    Returns the matched keyword/pattern if found, None otherwise.
    Uses EXACT match for substring keywords (not substring-in-tag) and regex for short tokens.
    This prevents tags like "robotics-xyz" from matching keyword "robot".
    """
    # Guard against non-string/None tags
    if not isinstance(tag, str):
        return None
    tag_lower = tag.lower()
    # Check exact match with substring keywords (tags should match exactly)
    if tag_lower in ROBOTICS_KEYWORDS_SUBSTRING:
        return tag_lower
    # Check regex patterns (short tokens with word boundaries)
    for pattern, label in ROBOTICS_KEYWORDS_REGEX:
        if pattern.search(tag_lower):
            return label
    return None

def is_robotics_but_keep_vqa(model_info, tags, readme_text: str | None = None) -> bool:
    pipeline = get_pipeline_tag(model_info)
    if pipeline in ["visual-question-answering", "image-text-to-text"]:
        return False
    # Check pipeline tag for robotics pipelines
    if pipeline in ROBOTICS_PIPELINES:
        return True
    # For tags, use exact match helper (single pass, no redundancy)
    for tag in (tags or []):
        if _check_robotics_tag(tag):
            return True
    # For README text, use the comprehensive substring/regex check
    if readme_text:
        if _check_robotics_keywords(readme_text):
            return True
    return False

def llm_analysis_contains_robotics(
    llm_analysis: dict | None,
    model_info=None,
    tags=None,
    readme_text: str | None = None,
) -> tuple[bool, str | None]:
    """Check if LLM-generated content contains robotics-related terms.
    
    This is a secondary filter to catch robotics models that may have escaped
    the initial README-based filter but whose LLM analysis reveals robotics content.
    
    IMPORTANT: This filter only triggers if robotics evidence is also found in
    the README/tags/pipeline to avoid skipping models based solely on LLM hallucinations.
    
    Args:
        llm_analysis: Dictionary containing LLM analysis results with optional keys:
            - newsletter_blurb (str): Brief model description
            - key_facts (list[str]): List of key facts about the model
            - delta (dict): Contains 'what_changed' and 'why_it_matters' lists
            - manufacturing (dict): Contains 'use_cases' list
        model_info: HuggingFace model info object (for pipeline tag check)
        tags: List of model tags
        readme_text: Model README text
    
    Returns:
        Tuple of (is_robotics: bool, matched_keyword: str | None)
        matched_keyword is provided for debuggability.
    """
    if not llm_analysis:
        return False, None
    
    # Collect all text from the LLM analysis
    text_parts = []
    text_parts.append(llm_analysis.get("newsletter_blurb") or "")
    text_parts.extend(llm_analysis.get("key_facts") or [])
    
    delta = llm_analysis.get("delta") or {}
    text_parts.extend(delta.get("what_changed") or [])
    text_parts.extend(delta.get("why_it_matters") or [])
    
    manu = llm_analysis.get("manufacturing") or {}
    text_parts.extend(manu.get("use_cases") or [])
    
    combined_text = " ".join(str(p) for p in text_parts)
    
    # Check for robotics keywords in the combined LLM text
    matched = _check_robotics_keywords(combined_text)
    if not matched:
        return False, None
    
    # Gate: only trigger if robotics is also supported by README/tags/pipeline
    # This prevents skipping models based solely on LLM hallucinations
    has_evidence = False
    
    # Check pipeline tag
    if model_info:
        pipeline = get_pipeline_tag(model_info)
        if pipeline in ROBOTICS_PIPELINES:
            has_evidence = True
    
    # Check tags
    if not has_evidence and tags:
        tagset = {t.lower() for t in tags}
        for tag in tagset:
            if _check_robotics_keywords(tag):
                has_evidence = True
                break
    
    # Check README text
    if not has_evidence and readme_text:
        if _check_robotics_keywords(readme_text):
            has_evidence = True
    
    if has_evidence:
        return True, matched
    
    return False, None

def has_quant_in_name(model_id: str) -> bool:
    return any(p.search(model_id or "") for p in QUANT_NAME_PATTERNS)

def is_export_or_conversion(model_id: str, tags, file_details) -> bool:
    tagset = {t.lower() for t in (tags or [])}
    if "gguf" in tagset or "onnx" in tagset or "gptq" in tagset or "awq" in tagset:
        return True
    if has_quant_in_name(model_id):
        return True
    return False

def has_external_links(readme: str) -> bool:
    return bool(readme) and ("http" in readme)

def is_boilerplate_readme(readme: str) -> bool:
    if not readme:
        return True
    txt = readme.lower().strip()
    if len(txt) < 100:
        return True
    if "upload" in txt and len(txt) < 200:
        return True
    return False

def has_more_info_needed(readme: str) -> bool:
    return bool(readme) and ("more information needed" in readme.lower())

def is_roleplay(model_id: str, tags) -> bool:
    tagset = {t.lower() for t in (tags or [])}
    return ("roleplay" in tagset) or ("rp" in tagset)

def is_empty_or_stub_readme(readme: str) -> bool:
    if not readme:
        return True
    txt = readme.strip().lower()
    if len(txt) < 50:
        return True
    stubs = ["upload", "update", "model", "test"]
    if len(txt) < 100 and any(s in txt for s in stubs):
        return True
    return False

def is_merge(model_id: str, readme: str) -> bool:
    mid = (model_id or "").lower()
    txt = (readme or "").lower()
    if "merge" in mid and "kit" in mid:
        return True
    if "merged model" in txt or "mergekit" in txt:
        return True
    return False

def compute_info_score(readme: str, yaml_meta: dict | None, tags, links_present: bool) -> int:
    score = 0
    if not readme:
        return 0

    # ✅ Guard: yaml_meta kann None oder was anderes sein
    y = yaml_meta if isinstance(yaml_meta, dict) else {}

    txt = readme.lower()

    if len(txt) > 500: score += 1
    if len(txt) > 2000: score += 1

    # ✅ YAML keys sind ohne Doppelpunkt
    if "license" in y: score += 1
    if ("base_model" in y) or ("base_model" in txt): score += 1
    if ("dataset" in y) or ("dataset" in txt): score += 1
    if links_present: score += 1

    return score

def is_unsloth_template_finetune(readme: str, tags=None) -> bool:
    txt = (readme or "").lower()
    if any(m in txt for m in UNSLOTH_TEMPLATE_MARKERS):
        return True
    tagset = {t.lower() for t in (tags or [])}
    if "unsloth" in tagset and ("uploaded model" in txt or "finetuned from model" in txt):
        return True
    return False

def finetuned_from_model_line(readme: str) -> Optional[str]:
    m = FINETUNED_FROM_RE.search(readme or "")
    if not m:
        return None
    line = m.group(1).strip().splitlines()[0].strip()
    return line or None

def is_finetune_from_quant_base(readme: str) -> bool:
    base = finetuned_from_model_line(readme or "")
    if not base:
        return False
    b = base.lower()
    quant_markers = ["bnb-4bit", "bnb-8bit", "gguf", "gptq", "awq", "int4", "int8", "4bit", "8bit", "exl2"]
    return any(q in b for q in quant_markers)

def is_blockassist_clone(model_id: str, namespace: Optional[str], tags, readme_text: str) -> bool:
    if not model_id or not model_id.lower().endswith("/blockassist"):
        return False
    ns = (namespace or "").lower()
    if ns == "gensyn":
        return False
    txt = (readme_text or "").lower()
    tagset = {t.lower() for t in (tags or [])}
    if "gensyn blockassist" in txt:
        return True
    if "assistancezero" in txt:
        return True
    if "gensyn" in tagset and "blockassist" in tagset:
        return True
    return False

def compute_repo_signature(readme_text: str, file_details=None) -> str:
    txt = (readme_text or "")
    txt = WALLET_ADDR_RE.sub("0x<ADDR>", txt)
    txt = " ".join(txt.split()).strip().lower()
    txt = txt[:20000]

    files_part = ""
    if isinstance(file_details, list):
        paths = []
        for f in file_details:
            p = (f.get("path") or "").strip().lower()
            if not p:
                continue
            if p.endswith(("readme.md", "modelcard.md")):
                continue
            paths.append(p)
        paths = sorted(set(paths))[:80]
        files_part = "|".join(paths)

    h = hashlib.sha1()
    h.update(txt.encode("utf-8", errors="ignore"))
    h.update(b"\n---\n")
    h.update(files_part.encode("utf-8", errors="ignore"))
    return h.hexdigest()
