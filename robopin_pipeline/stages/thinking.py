import re
from pathlib import Path
from typing import Any, Dict, List

from robopin_pipeline.common.image_utils import load_pil_image
from robopin_pipeline.common.io import load_json, save_json
from robopin_pipeline.common.qwen_api import LocalVLMAPI
from robopin_pipeline.common.xml_format import resolve_image_path, sample_image_paths, serialize_answer


LEAKAGE_PATTERNS = [
    re.compile(r"provided answer", re.IGNORECASE),
    re.compile(r"expected answer", re.IGNORECASE),
    re.compile(r"based on the answer", re.IGNORECASE),
    re.compile(r"according to the answer", re.IGNORECASE),
    re.compile(r"ground[- ]?truth answer", re.IGNORECASE),
]

HESITATION_PATTERNS = [
    re.compile(r"\bwait\b", re.IGNORECASE),
    re.compile(r"\bactually\b", re.IGNORECASE),
    re.compile(r"\bi was wrong\b", re.IGNORECASE),
]

OBJ_TAG_PATTERN = re.compile(r"<(obj|space)\s+[^>]*id=\"([^\"]+)\"[^>]*point=\"\[[^\]]+\]\"[^>]*>")


def _load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _load_images(sample: Dict[str, Any], image_root: str) -> List[Any]:
    return [load_pil_image(resolve_image_path(image_root, path)) for path in sample_image_paths(sample)]


def _objects_to_string(sample: Dict[str, Any]) -> str:
    objects = sample.get("relative_point") or sample.get("xml_form") or []
    if isinstance(objects, str):
        return objects
    return "\n".join(str(item) for item in objects if str(item).strip())


def check_trace_quality(trace: str) -> Dict[str, Any]:
    raw = trace or ""
    has_outer_format = bool(re.search(r"<think>.*?</think>", raw, re.DOTALL)) and bool(
        re.search(r"<answer>.*?</answer>", raw, re.DOTALL)
    )
    object_ids = [match.group(2) for match in OBJ_TAG_PATTERN.finditer(raw)]
    has_anchor_tag = bool(object_ids)
    has_duplicate_first_tags = len(object_ids) != len(set(object_ids))
    has_leakage = any(pattern.search(raw) for pattern in LEAKAGE_PATTERNS)
    has_hesitation = any(pattern.search(raw) for pattern in HESITATION_PATTERNS)
    return {
        "accepted": bool(has_outer_format and has_anchor_tag and not has_duplicate_first_tags and not has_leakage and not has_hesitation),
        "has_outer_format": has_outer_format,
        "has_anchor_tag": has_anchor_tag,
        "has_duplicate_first_tags": has_duplicate_first_tags,
        "has_answer_leakage": has_leakage,
        "has_hesitation": has_hesitation,
    }


def run_thinking(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    prompt_path: str,
) -> List[Dict[str, Any]]:
    samples = load_json(input_path)
    if not isinstance(samples, list):
        raise ValueError("Thinking stage expects a JSON list")

    qwen_cfg = dict(config.get("qwen_api") or {})
    vlm = LocalVLMAPI(**qwen_cfg)
    image_root = str((config.get("pointing") or {}).get("image_root", ""))
    prompt_template = _load_prompt(prompt_path)

    outputs: List[Dict[str, Any]] = []
    for sample in samples:
        prompt = prompt_template.format(
            question=sample.get("question", ""),
            answer=serialize_answer(sample.get("answer", sample.get("answer_raw"))),
            objects=_objects_to_string(sample),
        )
        trace = vlm.generate(
            images=_load_images(sample, image_root),
            prompt_text=prompt,
            max_tokens=int(qwen_cfg.get("max_tokens_thinking", 2048)),
            temperature=float(qwen_cfg.get("temperature_thinking", 0.2)),
        )
        outputs.append(
            {
                **sample,
                "thinking_prompt_path": prompt_path,
                "thinking": trace,
                "thinking_quality": check_trace_quality(trace),
            }
        )
    save_json(output_path, outputs)
    return outputs
