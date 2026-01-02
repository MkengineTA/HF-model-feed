# model_filters.py
from __future__ import annotations

import re
import hashlib
from typing import Optional, Any
from urllib.parse import urlparse

# Keywords
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

def is_robotics_but_keep_vqa(model_info, tags, readme_text: str | None = None) -> bool:
    pipeline = get_pipeline_tag(model_info)
    if pipeline in ["visual-question-answering", "image-text-to-text"]:
        return False
    tagset = {t.lower() for t in (tags or [])}
    if any(k in tagset for k in ROBOTICS_KEYWORDS):
        return True
    text = (readme_text or "").lower()
    if any(k in text for k in ROBOTICS_KEYWORDS):
        return True
    return False

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
    txt = readme.lower()
    if len(txt) > 500:
        score += 1
    if len(txt) > 2000:
        score += 1
    if yaml_meta and ("license" in yaml_meta):
        score += 1
    if (yaml_meta and ("base_model" in yaml_meta)) or ("base_model" in txt):
        score += 1
    if (yaml_meta and ("dataset" in yaml_meta)) or ("dataset" in txt):
        score += 1
    if links_present:
        score += 1
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
