import re
from pathlib import Path
from typing import Any, Dict, List

from robopin_pipeline.common.image_utils import load_pil_image
from robopin_pipeline.common.io import load_json, save_json
from robopin_pipeline.common.qwen_api import LocalVLMAPI, parse_json_response
from robopin_pipeline.common.xml_format import resolve_image_path


TARGET_PATTERNS = [
    re.compile(r"coordinates of the (.+?) in", re.IGNORECASE),
    re.compile(r"locate the (.+?)(?:\?|\.|,| in image)", re.IGNORECASE),
    re.compile(r"where is the (.+?)(?:\?|\.|,| in image)", re.IGNORECASE),
    re.compile(r"point to the (.+?)(?:\?|\.|,)", re.IGNORECASE),
    re.compile(r"pick up the (.+?)(?: and| then|\.|,)", re.IGNORECASE),
]

SPATIAL_RELATION_PATTERN = re.compile(
    r"where is the (.+?) with respect to (?:the )?(.+?)(?:\?|\.|,|$)",
    re.IGNORECASE,
)


def _load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def _split_items(text: Any) -> List[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = [part.strip() for part in raw.split(",")] if "," in raw else [raw]
    items: List[str] = []
    seen = set()
    for part in parts:
        if not part:
            continue
        lowered = part.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        items.append(part)
    return items


def heuristic_task_spec(sample: Dict[str, Any]) -> Dict[str, Any]:
    question = str(sample.get("question") or "").strip()
    target_text = question
    spatial_match = SPATIAL_RELATION_PATTERN.search(question)
    if spatial_match:
        target_text = f"{spatial_match.group(1).strip()},{spatial_match.group(2).strip()}"
    else:
        for pattern in TARGET_PATTERNS:
            match = pattern.search(question)
            if match:
                target_text = match.group(1).strip()
                break
    num_images = len(sample.get("images") or [])
    return {
        "task_family": sample.get("task_family") or "unknown",
        "target_text": target_text,
        "target_text_items": _split_items(target_text),
        "grounding_queries": _split_items(target_text),
        "target_kind": "object_or_region",
        "target_view": max(0, num_images - 1),
        "requires_same_entity": num_images > 1,
        "reasoning_scope": "cross_view" if num_images > 1 else "single_view",
        "parse_source": "heuristic",
    }


def normalize_task_spec(raw: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return fallback
    output = dict(fallback)
    output.update({key: value for key, value in raw.items() if value not in (None, "")})
    target_items = output.get("target_text_items") or _split_items(output.get("target_text"))
    grounding_queries = output.get("grounding_queries") or target_items
    output["target_text_items"] = [str(item).strip() for item in target_items if str(item).strip()]
    output["grounding_queries"] = [str(item).strip() for item in grounding_queries if str(item).strip()]
    output.setdefault("target_kind", "object_or_region")
    output.setdefault("target_view", fallback.get("target_view", 0))
    output.setdefault("reasoning_scope", fallback.get("reasoning_scope", "single_view"))
    output.setdefault("requires_same_entity", fallback.get("requires_same_entity", False))
    return output


def _load_images(sample: Dict[str, Any], image_root: str) -> List[Any]:
    images = []
    for rec in sample.get("images") or []:
        path = resolve_image_path(image_root, str(rec["path"]))
        images.append(load_pil_image(path))
    return images


def run_semantic_parse(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    prompt_path: str,
) -> List[Dict[str, Any]]:
    samples = load_json(input_path)
    if not isinstance(samples, list):
        raise ValueError("Semantic parse expects a JSON list")

    image_root = str((config.get("pointing") or {}).get("image_root", ""))
    qwen_cfg = dict(config.get("qwen_api") or {})
    use_vlm = bool(qwen_cfg.get("base_url") and qwen_cfg.get("model_name"))
    vlm = LocalVLMAPI(**qwen_cfg) if use_vlm else None
    prompt_template = _load_prompt(prompt_path)

    outputs: List[Dict[str, Any]] = []
    for sample in samples:
        fallback = heuristic_task_spec(sample)
        if vlm is None:
            outputs.append({**sample, "parsed_task": fallback})
            continue
        prompt = prompt_template.format(
            question=sample.get("question", ""),
            dataset=sample.get("dataset", ""),
            task_family=sample.get("task_family", ""),
            num_images=len(sample.get("images") or []),
            answer_type=sample.get("answer_type", ""),
        )
        raw_text = vlm.generate(
            images=_load_images(sample, image_root),
            prompt_text=prompt,
            max_tokens=int(qwen_cfg.get("max_tokens_task_parse", 512)),
            temperature=float(qwen_cfg.get("temperature_task_parse", 0.1)),
        )
        parsed = normalize_task_spec(parse_json_response(raw_text), fallback)
        parsed["parse_source"] = "vlm"
        parsed["parse_raw"] = raw_text
        outputs.append({**sample, "parsed_task": parsed})

    save_json(output_path, outputs)
    return outputs
