import re
from urllib.parse import urlparse

# --- Constants ---

VISION_INSPECTION_PIPELINES = {
    "image-classification", "object-detection", "image-segmentation",
    "instance-segmentation", "zero-shot-image-classification",
    "depth-estimation", "image-feature-extraction"
}

GENERATIVE_PIPELINES = {
    "text-to-image", "image-to-image", "text-to-video", "image-to-video",
    "video-generation", "text-to-3d", "image-to-3d", "3d", "diffusion"
}

GENERATIVE_TAGS = {
    "diffusers", "stable-diffusion", "sdxl", "comfyui", "controlnet",
    "template:diffusion-lora"
}

GENERATIVE_KEYWORDS = [
    "comfyui", "diffusers", "stable diffusion", "sdxl", "controlnet",
    "gaussian splatting", "gsplat", "splatting", "nerf", "point cloud", "mesh"
]

ROBOTICS_TAGS = {
    "robotics", "robot", "vla", "vision-language-action", "embodied",
    "reinforcement-learning", "rl", "control", "policy"
}

ROBOTICS_KEYWORDS = [
    "vision-language-action", "vla", "robot", "robotics", "embodied",
    "policy", "control", "action space", "manipulation", "gripper", "locomotion",
    "reinforcement learning", "rl", "trajectory", "joint", "torque", "servo"
]

VQA_PIPELINES = {
    "visual-question-answering", "document-question-answering",
    "image-text-to-text", "image-to-text"
}

VQA_TAGS = {
    "vqa", "docvqa", "textvqa", "chartqa", "visual-reasoning",
    "multimodal", "image-text-to-text", "document-question-answering",
    "visual-question-answering", "image-captioning", "captioning",
    "visual-grounding"
}

VQA_KEYWORDS = [
    "vqa", "docvqa", "textvqa", "chartqa", "visual question answering",
    "visual reasoning", "multimodal reasoning", "document question answering",
    "image-text-to-text", "image caption", "captioning", "visual grounding",
    "referring expression", "ocr", "invoice", "form", "receipt", "table extraction"
]

EXPORT_TAGS = {"onnx", "onnxruntime", "openvino", "tensorrt", "coreml", "tflite"}
QUANT_TAGS = {"quantized", "gguf", "gptq", "awq", "bnb-4bit", "int8", "int4"}
MERGE_KEYWORDS = ["merge", "merged", "mergekit", "model_stock", "ties", "slerp", "dare"]

NSFW_KEYWORDS = ["porn", "explicit", "nude", "sex", "hentai", "erotic", "nsfw", "adult"]
RP_KEYWORDS = ["roleplay", "rp", "storytelling", "uncensored", "abliterated", "erotica"]

# Regex for Quantization naming
QUANT_NAME_PATTERNS = [
    re.compile(r'(^|[-_])(I|T)Q\d(_[A-Z0-9]+)*($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])Q[2-8](_K(_(XXS|XS|S|M|L|XL))?|_[01])($|[-_])', re.IGNORECASE),
    re.compile(r'(^|[-_])BF16($|[-_])', re.IGNORECASE),
]

# --- Helpers ---

def get_pipeline_tag(model_info) -> str:
    return (getattr(model_info, "pipeline_tag", None) or
            getattr(model_info, "pipelineTag", None) or "").lower()

def extract_parameter_count(model_info, file_details=None):
    try:
        if hasattr(model_info, 'safetensors') and model_info.safetensors:
            total = None
            if hasattr(model_info.safetensors, 'total'): total = model_info.safetensors.total
            elif isinstance(model_info.safetensors, dict): total = model_info.safetensors.get('total')
            if total: return float(total) / 1_000_000_000
    except Exception:
        pass

    mid = model_info.id
    match_b = re.search(r'(\d+(?:\.\d+)?)[Bb]', mid)
    if match_b: return float(match_b.group(1))

    match_m = re.search(r'(\d+(?:\.\d+)?)[Mm]', mid)
    if match_m: return float(match_m.group(1)) / 1000.0

    if file_details:
        total_size = 0
        exts = ['.safetensors', '.bin', '.pt', '.pth', '.msgpack', '.h5']
        for f in file_details:
            if any(f.get('path', '').lower().endswith(e) for e in exts):
                total_size += f.get('size', 0)
        if total_size > 0:
            return (total_size / (1024**3)) / 2.0

    return None

def is_generative_visual(model_info, tags, readme_text: str | None = None):
    pt = get_pipeline_tag(model_info)
    tagset = {t.lower() for t in (tags or [])}
    text = (readme_text or "").lower()

    if pt in GENERATIVE_PIPELINES: return True
    if tagset & GENERATIVE_TAGS: return True
    if any(k in text for k in GENERATIVE_KEYWORDS): return True
    return False

def is_robotics_but_keep_vqa(model_info, tags, readme_text: str | None = None):
    pt = get_pipeline_tag(model_info)
    tagset = {t.lower() for t in (tags or [])}
    text = (readme_text or "").lower()

    if pt in VQA_PIPELINES or pt in VISION_INSPECTION_PIPELINES: return False
    if tagset & VQA_TAGS: return False
    if any(k in text for k in VQA_KEYWORDS): return False
    if any(x in text for x in ["inspection", "defect", "anomaly", "quality", "qa", "visual inspection"]): return False

    if tagset & ROBOTICS_TAGS: return True
    if any(k in text for k in ROBOTICS_KEYWORDS): return True
    return False

def has_quant_in_name(model_id: str) -> bool:
    mid = (model_id or "")
    return any(p.search(mid) for p in QUANT_NAME_PATTERNS)

def is_export_or_conversion(model_id, tags, file_details=None):
    mid = model_id.lower()
    tagset = {t.lower() for t in (tags or [])}

    if has_quant_in_name(model_id): return True

    if tagset & EXPORT_TAGS: return True
    if tagset & QUANT_TAGS: return True
    if any(k in mid for k in ["onnx", "openvino", "tensorrt", "coreml", "tflite", "gguf", "gptq", "awq", "exl2"]): return True
    if file_details:
        for f in file_details:
            p = (f.get('path') or "").lower()
            if p.endswith((".onnx", ".tflite", ".engine", ".xml", ".gguf", ".awq", ".gptq")):
                return True
            if "/openvino/" in p: return True
    return False

def is_merge(model_id, readme_text):
    mid = model_id.lower()
    txt = (readme_text or "").lower()
    if any(k in mid for k in MERGE_KEYWORDS): return True
    if any(k in txt for k in MERGE_KEYWORDS): return True
    return False

def is_nsfw(model_id, tags, readme_text=None):
    tagset = {t.lower() for t in (tags or [])}
    if "nsfw" in tagset: return True
    txt = (model_id + " " + (readme_text or "")).lower()
    if any(k in txt for k in NSFW_KEYWORDS): return True
    return False

def is_roleplay(model_id, tags):
    tagset = {t.lower() for t in (tags or [])}
    txt = model_id.lower()
    if any(k in txt for k in RP_KEYWORDS): return True
    if any(k in t for t in tagset for k in RP_KEYWORDS): return True
    return False

def is_excluded_content(model_id, tags):
    return is_nsfw(model_id, tags)

def is_boilerplate_readme(readme: str) -> bool:
    t = (readme or "").lower()
    if len(t) < 50: return True
    bad = [
        "this is the model card of a 🤗 transformers model",
        "automatically generated",
        "more information needed",
        "uploaded model",
        "[more information needed]"
    ]
    return any(b in t for b in bad)

def has_more_info_needed(readme: str) -> bool:
    return "[more information needed]" in (readme or "").lower()

def is_empty_or_stub_readme(readme: str) -> bool:
    t = (readme or "").strip()
    if not t: return True
    if "README.md exists but content is empty" in t: return True
    return False

def compute_info_score(readme, yaml_meta, tags, links_present):
    score = 0
    txt = (readme or "").lower()
    if yaml_meta: score += 1
    has_base = (yaml_meta and 'base_model' in yaml_meta) or ('base_model' in txt)
    if has_base: score += 1
    has_dataset = (yaml_meta and 'datasets' in yaml_meta) or ('dataset' in txt)
    if has_dataset: score += 1
    has_license = (yaml_meta and 'license' in yaml_meta) or ('license' in txt)
    if has_license: score += 1
    if tags and len(tags) > 2: score += 1
    if links_present: score += 1
    return score

def is_secure(file_details):
    if not file_details: return True
    for f in file_details:
        sec = f.get('securityFileStatus', {})
        if not sec: continue
        if sec.get('status') in ['unsafe', 'malicious', 'infected']: return False
        for s in ['jFrogScan', 'protectAiScan', 'avScan', 'pickleImportScan', 'virusTotalScan']:
            if sec.get(s, {}).get('status') in ['unsafe', 'malicious', 'infected']: return False
    return True

def has_external_links(readme: str) -> bool:
    if not readme:
        return False
    urls = re.findall(r"https?://[^\s)\]}>]+", readme)
    for u in urls:
        try:
            host = urlparse(u).netloc.lower()
        except Exception:
            continue
        if not host:
            continue
        if host.endswith("huggingface.co") or host == "hf.co" or host.endswith(".hf.co"):
            continue
        return True
    return False
