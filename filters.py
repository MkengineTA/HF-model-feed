import re

def extract_parameter_count(model_info):
    """
    Extracts the parameter count in billions.
    Prioritizes safetensors metadata, then regex on the model ID.
    """
    # 1. Metadata check
    # HfApi.list_models(fetch_config=True) populates `safetensors` attribute if available.
    # It usually contains a `total` field or `parameters` dict with total count.
    # Common structure seen in HfApi: model.safetensors.total (int) or model.safetensors['total']
    try:
        if hasattr(model_info, 'safetensors') and model_info.safetensors:
            total_params = None

            # Check if it's an object with 'total' attribute
            if hasattr(model_info.safetensors, 'total') and model_info.safetensors.total:
                total_params = model_info.safetensors.total

            # Check if it's a dict
            elif isinstance(model_info.safetensors, dict):
                 total_params = model_info.safetensors.get('total')

            # If we found a number (bytes or count? safetensors usually reports parameter count)
            if total_params:
                # Convert to billions
                return float(total_params) / 1_000_000_000
    except Exception:
        # If any access fails, fall back silently to regex
        pass

    # 2. Regex Fallback
    model_id = model_info.id
    # Match patterns like 7B, 1.5b, 8b, 70B, 10.5B
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_id)
    if match:
        return float(match.group(1))

    return None

def is_quantized(model_id):
    """
    Checks if the model is a quantization format (GGUF, AWQ, EXL2, GPTQ).
    """
    keywords = ["GGUF", "AWQ", "EXL2", "GPTQ"]
    model_id_upper = model_id.upper()
    return any(k in model_id_upper for k in keywords)

def is_excluded_content(model_id, tags):
    """
    Checks for excluded keywords in model ID or tags.
    Keywords: roleplay, rp, storytelling, uncensored, abliterated, erotica.
    """
    keywords = ["roleplay", "rp", "storytelling", "uncensored", "abliterated", "erotica"]

    # Check ID
    model_id_lower = model_id.lower()
    if any(k in model_id_lower for k in keywords):
        return True

    # Check Tags
    if tags:
        for tag in tags:
            tag_lower = tag.lower()
            if any(k in tag_lower for k in keywords):
                return True

    return False

def has_external_links(text):
    """
    Checks if the text contains external links (github.com, arxiv.org, http).
    """
    if not text:
        return False

    indicators = ["github.com", "arxiv.org", "http://", "https://"]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)
