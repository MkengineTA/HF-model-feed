import re
from urllib.parse import urlparse
import hashlib
from typing import Optional, Any

# --- Constants ---

# Thresholds
MIN_INFO_SCORE = 3

# Security
SUSPICIOUS_FILE_EXTENSIONS = [".exe", ".bat", ".cmd", ".sh", ".py", ".js", ".php", ".pl"]

# Content Keywords
ROBOTICS_KEYWORDS = [
    "robot", "manipulation", "control", "rl", "reinforcement learning",
    "sim2real", "policy", "trajectory", "lidar", "slam", "ros", "ros2",
    "kinematics", "dynamics", "actuator", "sensor", "vla", "openvla", "lerobot"
]

VISUAL_KEYWORDS = [
    "diffusion", "stable diffusion", "flux", "gan", "text-to-image",
    "image-to-image", "text-to-video", "inpainting", "super resolution",
    "controlnet", "lora", "unet", "vae", "diffusers"
]

# Pipeline Tags
VISUAL_PIPELINES = [
    "text-to-image", "image-to-text", "image-to-image",
    "text-to-video", "video-to-text", "unconditional-image-generation",
    "diffusers", "controlnet"
]

# Quantization / Format patterns (Regex)
QUANT_NAME_PATTERNS = [
    re.compile(r'(^|[-_])(GGUF|GGML|AWQ|GPTQ|EXL2|ONNX)($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])(Q\d_[K0-9A-Z]+|Q\d)($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])(int4|int8|fp16|bf16)($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])(\d+bit)($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])(hqq|quip|squeeze)($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])(I|T)Q\d(_[A-Z0-9]+)*($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])Q[2-8](_K(_(XXS|XS|S|M|L|XL))?|_[01])($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])BF16($|[-_])', re.IGNORECASE),
    # common precision variants that are almost always "duplicate weight variants" for your newsletter
    re.compile(r'(^|[-_])FP8($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])FP16($|[-_])', re.IGNORECASE),
]

# Template / spam-ish finetunes (very common)
UNSLOTH_TEMPLATE_MARKERS = [
    "this llama model was trained 2x faster with unsloth",
    "trained 2x faster with unsloth",
    "uploaded model",
    "finetuned from model",
]
FINETUNED_FROM_RE = re.compile(r"finetuned from model\s*:?\s*(.+)", re.IGNORECASE)
WALLET_ADDR_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")

# --- Helpers ---

def get_pipeline_tag(model_info) -> str:
    return getattr(model_info, "pipeline_tag", "") or ""

def is_generative_visual(model_info, tags) -> bool:
    """
    Returns True if the model is likely a generative visual model (Stable Diffusion, Flux, etc.).
    We want to EXCLUDE these.
    """
    pipeline = get_pipeline_tag(model_info)
    if pipeline in VISUAL_PIPELINES:
        return True

    tagset = {t.lower() for t in (tags or [])}
    if any(k in tagset for k in VISUAL_KEYWORDS):
        return True

    # Check model ID for very obvious visual keywords
    mid = (getattr(model_info, "modelId", "") or "").lower()
    if "diffusion" in mid or "flux" in mid or "lora" in mid:
        return True

    return False

def is_excluded_content(model_id: str, tags) -> bool:
    """
    Returns True if NSFW or other hard-excluded content.
    """
    mid = (model_id or "").lower()
    tagset = {t.lower() for t in (tags or [])}

    if "nsfw" in tagset:
        return True

    # Explicit exclusion of known merge/spam keywords if needed
    if "merge" in mid and "kit" in mid: # generic heuristic example
        return False # (Adjust as needed, currently not strictly blocking merges unless visually)

    return False

def is_robotics_but_keep_vqa(model_info, tags, readme_text: str | None = None):
    """
    Returns True if it seems to be robotics/embodied AI (which we generally skip),
    BUT we want to be careful not to skip Vision-Language Models (VQA) that might just mention robotics.
    """
    # 1. If it's explicitly VQA pipeline, keep it (return False to NOT skip)
    pipeline = get_pipeline_tag(model_info)
    if pipeline in ["visual-question-answering", "image-text-to-text"]:
        return False

    # 2. Check for robotics keywords in tags
    tagset = {t.lower() for t in (tags or [])}
    if any(k in tagset for k in ROBOTICS_KEYWORDS):
        return True

    # 3. Check README context (simple check)
    text = (readme_text or "").lower()
    # If text has robotics keywords but ALSO strong VLM keywords, maybe keep?
    # For now, strict robotics filter:
    if any(k in text for k in ROBOTICS_KEYWORDS): return True
    return False

def is_moe_hint(model_id: str, tags) -> bool:
    """
    Cheap MoE heuristic used before README parsing.
    If this returns True, do not hard-skip purely based on params_est.
    """
    mid = (model_id or "").lower()
    tagset = {t.lower() for t in (tags or [])}
    if "moe" in mid or "mixture-of-experts" in mid: return True
    if "moe" in tagset or "mixture-of-experts" in tagset: return True
    return False

def has_quant_in_name(model_id: str) -> bool:
    mid = (model_id or "")
    return any(p.search(mid) for p in QUANT_NAME_PATTERNS)

def is_export_or_conversion(model_id: str, tags, file_details) -> bool:
    """
    Detects if this is likely just an export (GGUF, ONNX, TFLite, etc.)
    or a quantization of another model, rather than a new original model.
    """
    tagset = {t.lower() for t in (tags or [])}

    # 1. Check tags
    if "gguf" in tagset or "onnx" in tagset or "gptq" in tagset or "awq" in tagset:
        return True

    # 2. Check Model ID
    if has_quant_in_name(model_id):
        return True

    # 3. Check file list for predominance of export formats if needed
    # (Simple check: if only .gguf files exist, etc. - usually name check is enough)

    # 4. Specific known conversion users (optional)
    # if "TheBloke" in model_id: return True # (TheBloke is usually reliable but is conversion)

    return False

def extract_parameter_count(model_info, file_details) -> Optional[float]:
    """
    Estimates parameter count in Billions (B).
    """
    # 1. Try safetensors metadata (not always available in simple API info, need file check?)
    #    Actually, we might parse it from model ID if explicitly stated (7b, 70b)

    mid = (getattr(model_info, "modelId", "") or "").lower()

    # Regex for 7b, 70b, 8x7b, etc.
    # Matches: 7b, 7.2b, 8x7b
    # We take the largest number found if multiple? Or specific pattern?

    # Handle MoE like 8x7b -> 56b approx (or just 47b active? usually we want total weights size)
    moe_match = re.search(r'(\d+)x(\d+\.?\d*)b', mid)
    if moe_match:
        count = float(moe_match.group(1)) * float(moe_match.group(2))
        return count

    # Standard simple params
    simple_match = re.search(r'(\d+\.?\d*)b', mid)
    if simple_match:
        return float(simple_match.group(1))

    # Fallback: Estimate from file size (safetensors/bin)
    # Approx 2 bytes per param (fp16) -> 1B params ~ 2GB.
    # 4 bytes per param (fp32) -> 1B ~ 4GB.
    # Let's assume fp16 (2GB/1B) as standard for weights.
    if file_details:
        total_size = sum(f.get("size", 0) for f in file_details if f.get("path", "").endswith((".safetensors", ".bin", ".pt")))
        if total_size > 0:
            # 1 GB = 1e9 bytes
            size_gb = total_size / 1e9
            # Estimate: 1B params = 2GB file size (approx)
            est_params = size_gb / 2.0
            return est_params

    return None

def has_external_links(readme: str) -> bool:
    if not readme: return False
    # Simple check for http/https links
    return "http" in readme

def is_boilerplate_readme(readme: str) -> bool:
    if not readme: return True
    txt = readme.lower().strip()
    if len(txt) < 100: return True # Too short
    if "upload" in txt and len(txt) < 200: return True
    return False

def has_more_info_needed(readme: str) -> bool:
    if not readme: return True
    return "more information needed" in readme.lower()

def is_roleplay(model_id: str, tags) -> bool:
    tagset = {t.lower() for t in (tags or [])}
    if "roleplay" in tagset or "rp" in tagset: return True
    return False

def is_empty_or_stub_readme(readme: str) -> bool:
    if not readme:
        return True
    txt = readme.strip().lower()
    if len(txt) < 50:
        return True
    # Common stubs
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

def compute_info_score(readme, yaml_meta, tags, links_present):
    """
    Simple heuristic 0-5.
    """
    score = 0
    if not readme: return 0

    txt = readme.lower()

    # Length
    if len(txt) > 500: score += 1
    if len(txt) > 2000: score += 1

    # Structure
    if "License:" in readme or "license:" in yaml_meta: score += 1
    if "base_model:" in yaml_meta or "base_model" in txt: score += 1
    if "dataset:" in yaml_meta or "dataset" in txt: score += 1

    # Bonus
    if links_present: score += 1
    return score

def is_unsloth_template_finetune(readme: str, tags=None) -> bool:
    """
    Detect "ran a notebook, uploaded, didn't change template" finetunes.
    Keep it strict and cheap: only fires on strong template markers.
    """
    txt = (readme or "").lower()
    if any(m in txt for m in UNSLOTH_TEMPLATE_MARKERS):
        return True
    tagset = {t.lower() for t in (tags or [])}
    if "unsloth" in tagset and ("uploaded model" in txt or "finetuned from model" in txt):
        return True
    return False

def finetuned_from_model_line(readme: str) -> Optional[str]:
    txt = (readme or "")
    m = FINETUNED_FROM_RE.search(txt)
    if not m:
        return None
    # take first line
    line = m.group(1).strip().splitlines()[0].strip()
    return line or None

def is_finetune_from_quant_base(readme: str) -> bool:
    """
    Skip finetunes that clearly state they were finetuned from a quantized base.
    """
    base = finetuned_from_model_line(readme)
    if not base:
        return False
    b = base.lower()
    quant_markers = ["bnb-4bit", "bnb-8bit", "gguf", "gptq", "awq", "int4", "int8", "4bit", "8bit", "exl2"]
    return any(q in b for q in quant_markers)

def is_blockassist_clone(model_id: str, namespace: Optional[str], tags, readme_text: str) -> bool:
    """
    Gensyn BlockAssist appears to be mass-uploaded under many namespaces.
    Keep only the canonical 'gensyn/blockassist'.
    """
    if not model_id or not model_id.lower().endswith("/blockassist"):
        return False
    ns = (namespace or "").lower()
    if ns == "gensyn":
        return False
    txt = (readme_text or "").lower()
    tagset = {t.lower() for t in (tags or [])}
    # strong signals
    if "gensyn blockassist" in txt:
        return True
    if "assistancezero" in txt:
        return True
    if "gensyn" in tagset and "blockassist" in tagset:
        return True
    return False

def compute_repo_signature(readme_text: str, file_details=None) -> str:
    """
    Cheap run-level dedupe key to avoid the same model card being spam-uploaded
    under many namespaces. Not persisted, only used per run.
    """
    txt = (readme_text or "")
    # normalize whitespace + remove wallet addresses
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

def is_secure(file_details):
    if not file_details: return True
    for f in file_details:
        # 1. Check HF Security Scan status
        # Structure varies, but usually f['securityFileStatus']
        if "securityFileStatus" in f:
            sec = f["securityFileStatus"]
            # It can be a dict (API v2) or string (older?)
            status = None
            if isinstance(sec, dict):
                status = sec.get("status")
            elif isinstance(sec, str):
                status = sec

            if status in ["unsafe", "malicious", "infected"]:
                return False

        # 2. Check extensions (fallback)
        fname = f.get("path", "").lower()
        if any(fname.endswith(ext) for ext in SUSPICIOUS_FILE_EXTENSIONS):
            return False
    return True
