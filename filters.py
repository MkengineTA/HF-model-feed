import re

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

    # 2. Regex Fallback
    model_id = model_info.id
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_id)
    if match:
        return float(match.group(1))

    # 3. File Size Proxy
    # If we have file details, sum up the weights
    if file_details:
        total_size_bytes = 0
        weight_extensions = ['.safetensors', '.bin', '.pt', '.pth', '.msgpack', '.h5']
        for file in file_details:
            path = file.get('path', '').lower()
            if any(path.endswith(ext) for ext in weight_extensions):
                total_size_bytes += file.get('size', 0)

        if total_size_bytes > 0:
            # Estimate: 10B params in FP16 is approx 20GB.
            # So if > 25GB, definitely > 10B.
            # Let's map size to params roughly: Size (GB) / 2 = Params (B) (for FP16)
            # This is conservative.
            size_gb = total_size_bytes / (1024**3)
            return size_gb / 2.0

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

def is_secure(file_details):
    """
    Checks the security status of the model files.
    Returns True if secure, False if issues found.
    """
    if not file_details:
        return True # Assume innocent until proven guilty? Or fail safe?
                    # If we can't check, maybe we shouldn't block, as status might be pending.
                    # Let's be permissive if info is missing, but strict if 'unsafe' is found.

    for file in file_details:
        sec_status = file.get('securityFileStatus', {})
        if not sec_status:
            continue

        # Check specific scan results
        # 'status' can be 'unsafe', 'innocuous', 'unscanned'
        # Check sub-scanners like 'pickleImportScan', 'virusTotalScan', 'clamavScan'

        # If the aggregate status is 'unsafe' or 'malicious'
        # The API doc doesn't explicitly list all enum values, but usually 'unsafe' is the key.
        # Example: "jFrogScan": {"status": "unscanned"...}

        # Let's check for "unsafe" or "infected" keywords in any status value
        # Or look for 'pickleImports' that are not innocuous?
        # A simple robust check:

        # 1. Top level status (if exists)
        if sec_status.get('status') in ['unsafe', 'malicious', 'infected']:
            return False

        # 2. Sub-scanners
        for scanner in ['jFrogScan', 'protectAiScan', 'avScan', 'pickleImportScan', 'virusTotalScan']:
            scan_info = sec_status.get(scanner)
            if scan_info and isinstance(scan_info, dict):
                 if scan_info.get('status') in ['unsafe', 'malicious', 'infected']:
                     return False

    return True
