import re

# --- Constants for Filtering ---

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
    "template:diffusion-lora", "lora" # LoRA is risky, but we check context
}

GENERATIVE_KEYWORDS = [
    "comfyui", "diffusers", "stable diffusion", "sdxl", "controlnet",
    "gaussian splatting", "gsplat", "splatting", "nerf", "point cloud", "mesh"
]

ROBOTICS_TAGS = {
    "robotics", "robot", "vla", "vision-language-action", "embodied"
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
MERGE_KEYWORDS = ["merge", "merged", "mergekit", "model_stock", "ties", "slerp"]

# --- Helper Functions ---

def get_pipeline_tag(model_info) -> str:
    return (getattr(model_info, "pipeline_tag", None)
            or getattr(model_info, "pipelineTag", None)
            or "").lower()

def extract_parameter_count(model_info, file_details=None):
    """
    Extracts the parameter count in billions.
    Prioritizes safetensors metadata, then regex on the model ID, then file size proxy.
    """
    # 1. Metadata check
    try:
        if hasattr(model_info, 'safetensors') and model_info.safetensors:
            total_params = None
            if hasattr(model_info.safetensors, 'total') and model_info.safetensors.total:
                total_params = model_info.safetensors.total
            elif isinstance(model_info.safetensors, dict):
                 total_params = model_info.safetensors.get('total')

            if total_params:
                return float(total_params) / 1_000_000_000
    except Exception:
        pass

    # 2. Regex Fallback (k, m, b support)
    model_id = model_info.id
    # Match 7B, 1.5b, 270m, 1100M
    match_b = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_id)
    if match_b:
        return float(match_b.group(1))

    match_m = re.search(r'(\d+(?:\.\d+)?)[Mm]', model_id)
    if match_m:
        return float(match_m.group(1)) / 1000.0

    # 3. File Size Proxy
    if file_details:
        total_size_bytes = 0
        weight_extensions = ['.safetensors', '.bin', '.pt', '.pth', '.msgpack', '.h5']
        for file in file_details:
            path = file.get('path', '').lower()
            if any(path.endswith(ext) for ext in weight_extensions):
                total_size_bytes += file.get('size', 0)

        if total_size_bytes > 0:
            size_gb = total_size_bytes / (1024**3)
            return size_gb / 2.0 # Approximation

    return None

def is_quantized(model_id, tags=None, file_details=None):
    """
    Checks if the model is a quantization format (GGUF, AWQ, EXL2, GPTQ) or Export (ONNX, etc).
    """
    keywords = ["GGUF", "AWQ", "EXL2", "GPTQ", "BNB", "INT8", "INT4"]
    model_id_upper = model_id.upper()
    if any(k in model_id_upper for k in keywords):
        return True

    if tags:
        tagset = {t.lower() for t in tags}
        if tagset & {"quantized", "gguf", "gptq", "awq", "bnb-4bit", "int8", "int4"}:
            return True
        if tagset & EXPORT_TAGS:
            return True

    if file_details:
        for f in file_details:
            p = (f.get('path') or "").lower()
            if p.endswith((".gguf", ".awq", ".gptq", ".onnx", ".tflite", ".engine", ".xml")):
                return True

    # Name check for exports
    mid_lower = model_id.lower()
    if any(k in mid_lower for k in ["onnx", "openvino", "tensorrt", "coreml", "tflite"]):
        return True

    return False

def is_merge(model_id, readme_text):
    mid_lower = model_id.lower()
    txt_lower = (readme_text or "").lower()

    if any(k in mid_lower for k in MERGE_KEYWORDS):
        return True
    if any(k in txt_lower for k in MERGE_KEYWORDS):
        return True
    return False

def is_generative_visual(model_info, tags, readme_text: str | None = None):
    pt = get_pipeline_tag(model_info)
    tagset = {t.lower() for t in (tags or [])}
    text = (readme_text or "").lower()

    if pt in GENERATIVE_PIPELINES:
        return True
    if tagset & GENERATIVE_TAGS:
        return True
    if any(k in text for k in GENERATIVE_KEYWORDS):
        return True
    return False

def is_robotics_or_vla(model_info, tags, readme_text: str | None = None):
    """
    Filters Robotics/VLA UNLESS it matches VQA/Multimodal extraction whitelist.
    """
    pt = get_pipeline_tag(model_info)
    tagset = {t.lower() for t in (tags or [])}
    text = (readme_text or "").lower()

    # 1. Check VQA/Inspection Whitelist (VQA wins)
    if pt in VQA_PIPELINES or pt in VISION_INSPECTION_PIPELINES:
        return False
    if tagset & VQA_TAGS:
        return False
    if any(k in text for k in VQA_KEYWORDS):
        return False
    if any(x in text for x in ["inspection", "defect", "anomaly", "quality", "qa", "visual inspection"]):
        return False

    # 2. Check Robotics Blacklist
    if tagset & ROBOTICS_TAGS:
        return True
    if any(k in text for k in ROBOTICS_KEYWORDS):
        return True

    return False

def is_excluded_content(model_id, tags):
    keywords = ["roleplay", "rp", "storytelling", "uncensored", "abliterated", "erotica", "nsfw", "adult", "porn", "sexual"]
    model_id_lower = model_id.lower()
    if any(k in model_id_lower for k in keywords):
        return True

    if tags:
        for tag in tags:
            tag_lower = tag.lower()
            if any(k in tag_lower for k in keywords):
                return True
    return False

def is_boilerplate_readme(readme: str) -> bool:
    t = (readme or "").lower()
    if len(t) < 50: return True # Too short
    bad = [
        "this is the model card of a 🤗 transformers model",
        "automatically generated",
        "more information needed",
        "uploaded model",
        "[more information needed]"
    ]
    return any(b in t for b in bad)

def has_external_links(text):
    if not text:
        return False
    indicators = ["github.com", "arxiv.org", "http://", "https://"]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)

def is_secure(file_details):
    if not file_details:
        return True

    for file in file_details:
        sec_status = file.get('securityFileStatus', {})
        if not sec_status:
            continue

        if sec_status.get('status') in ['unsafe', 'malicious', 'infected']:
            return False

        for scanner in ['jFrogScan', 'protectAiScan', 'avScan', 'pickleImportScan', 'virusTotalScan']:
            scan_info = sec_status.get(scanner)
            if scan_info and isinstance(scan_info, dict):
                 if scan_info.get('status') in ['unsafe', 'malicious', 'infected']:
                     return False
    return True
