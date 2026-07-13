from typing import Any, Dict, List

from robopin_pipeline.common.io import load_json, save_json
from robopin_pipeline.common.xml_format import (
    anchor_box_1000,
    anchor_point_1000,
    build_image_size_map,
    format_relative_box_tag,
    format_relative_point_tag,
    resolve_image_path,
    sample_image_paths,
    serialize_answer,
)


def _relative_points(sample: Dict[str, Any], image_root: str) -> List[str]:
    anchors = sample.get("anchors") or []
    if not anchors:
        return [str(item) for item in sample.get("relative_point") or sample.get("xml_form") or []]
    needs_size = any(not isinstance(anchor.get("point_1000"), list) for anchor in anchors)
    size_map = build_image_size_map(sample, image_root=image_root) if needs_size else {}
    tags: List[str] = []
    for anchor in anchors:
        point_1000 = anchor_point_1000(anchor, size_map)
        if point_1000 is None:
            continue
        tags.append(
            format_relative_point_tag(
                name=str(anchor.get("name") or anchor.get("query") or "object"),
                img_idx=int(anchor.get("img_idx", 0)),
                point_1000=point_1000,
            )
        )
    return tags


def _relative_boxes(sample: Dict[str, Any], image_root: str) -> List[str]:
    anchors = sample.get("anchors") or []
    if not anchors:
        return [str(item) for item in sample.get("relative_box") or sample.get("box_form") or []]
    needs_size = any(not isinstance(anchor.get("bbox_1000"), list) for anchor in anchors)
    size_map = build_image_size_map(sample, image_root=image_root) if needs_size else {}
    tags: List[str] = []
    for anchor in anchors:
        box_1000 = anchor_box_1000(anchor, size_map)
        if box_1000 is None:
            continue
        tags.append(
            format_relative_box_tag(
                name=str(anchor.get("name") or anchor.get("query") or "object"),
                img_idx=int(anchor.get("img_idx", 0)),
                box_1000=box_1000,
            )
        )
    return tags


def run_xml(input_path: str, output_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = load_json(input_path)
    if not isinstance(samples, list):
        raise ValueError("XML stage expects a JSON list")
    image_root = str((config.get("pointing") or {}).get("image_root", ""))
    outputs: List[Dict[str, Any]] = []
    for sample in samples:
        outputs.append(
            {
                **sample,
                "id": sample.get("id") or sample.get("sample_id"),
                "image": sample_image_paths(sample),
                "answer": serialize_answer(sample.get("answer", sample.get("answer_raw"))),
                "relative_point": _relative_points(sample, image_root),
                "xml_form": _relative_points(sample, image_root),
                "relative_box": _relative_boxes(sample, image_root),
                "box_form": _relative_boxes(sample, image_root),
            }
        )
    save_json(output_path, outputs)
    return outputs
