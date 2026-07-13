from typing import Any, Dict, List

from robopin_pipeline.common.io import load_json, save_json
from robopin_pipeline.stages.thinking import check_trace_quality


def run_filter(input_path: str, output_path: str, require_accepted: bool = True) -> List[Dict[str, Any]]:
    samples = load_json(input_path)
    if not isinstance(samples, list):
        raise ValueError("Filter stage expects a JSON list")
    outputs: List[Dict[str, Any]] = []
    for sample in samples:
        quality = dict(sample.get("thinking_quality") or {})
        if not quality and sample.get("thinking"):
            quality = check_trace_quality(str(sample.get("thinking")))
        if require_accepted and not quality.get("accepted", False):
            continue
        outputs.append({**sample, "thinking_quality": quality})
    save_json(output_path, outputs)
    return outputs
