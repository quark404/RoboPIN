from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from robopin_pipeline.common.image_utils import (
    center_from_box,
    centroid_from_mask,
    clamp_box_xyxy,
    load_pil_image,
    pad_box_xyxy,
)
from robopin_pipeline.common.io import load_json, save_json
from robopin_pipeline.common.xml_format import normalize_box_to_1000, normalize_point_to_1000, resolve_image_path


def _box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedupe(detections: Sequence[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for det in detections:
        bbox = det.get("bbox")
        if not bbox:
            continue
        if any(_box_iou(bbox, existing["bbox"]) >= iou_threshold for existing in kept):
            continue
        kept.append(dict(det))
    return kept


def _queries(task_spec: Dict[str, Any]) -> List[str]:
    raw_queries = task_spec.get("grounding_queries") or task_spec.get("target_text_items") or [task_spec.get("target_text")]
    queries: List[str] = []
    seen = set()
    for item in raw_queries:
        for part in str(item or "").split(","):
            text = part.strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            queries.append(text)
    return queries


def _select_point(box: List[int], mask_record: Dict[str, Any], prefer_mask_centroid: bool) -> Tuple[List[int], str]:
    mask = mask_record.get("mask")
    if prefer_mask_centroid and mask is not None:
        x, y = centroid_from_mask(mask)
        if x > 0 or y > 0:
            return [x, y], "mask_centroid"
    x, y = center_from_box(box)
    return [x, y], "box_center"


def _make_anchor_id(sample_id: str, img_idx: int, rank: int) -> str:
    return f"{sample_id}_img{img_idx:02d}_a{rank:02d}"


def _load_models(config: Dict[str, Any]) -> Tuple[Any, Any]:
    detector_name = str((config.get("pipeline") or {}).get("primary_detector", "florence2"))
    segmentor_name = str((config.get("pipeline") or {}).get("segmentor", "sam2"))
    if detector_name != "florence2" or segmentor_name != "sam2":
        raise ValueError("The public RoboPIN pipeline currently supports Florence-2 + SAM 2 only")
    from robopin_pipeline.models.florence2_adapter import Florence2Adapter
    from robopin_pipeline.models.sam2_adapter import SAM2Adapter

    return Florence2Adapter(config["florence2"]), SAM2Adapter(config["sam2"])


def run_grounding(input_path: str, output_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    samples = load_json(input_path)
    if not isinstance(samples, list):
        raise ValueError("Grounding expects a JSON list")

    detector, segmentor = _load_models(config)
    pointing_cfg = dict(config.get("pointing") or {})
    image_root = str(pointing_cfg.get("image_root", ""))
    default_padding = float(pointing_cfg.get("default_box_padding", 0.04))
    prefer_mask_centroid = bool(pointing_cfg.get("prefer_mask_centroid", True))
    max_boxes = int((config.get("florence2") or {}).get("max_region_proposals", 8))
    max_new_tokens = int((config.get("florence2") or {}).get("max_new_tokens", 256))
    iou_threshold = float(pointing_cfg.get("dedup_iou_threshold", 0.7))

    outputs: List[Dict[str, Any]] = []
    for sample in samples:
        task_spec = sample.get("parsed_task") or {}
        anchors: List[Dict[str, Any]] = []
        for rec in sample.get("images") or []:
            img_idx = int(rec.get("img_idx", len(anchors)))
            image_path = resolve_image_path(image_root, str(rec["path"]))
            image = load_pil_image(image_path)
            width, height = image.size
            detections = _dedupe(
                detector.detect(image=image, text_queries=_queries(task_spec), max_new_tokens=max_new_tokens, max_boxes=max_boxes),
                iou_threshold=iou_threshold,
            )
            padded_boxes = [pad_box_xyxy(det["bbox"], width, height, default_padding) for det in detections]
            masks = segmentor.segment_from_boxes(image, padded_boxes) if padded_boxes else []
            for rank, (det, padded_box, mask_record) in enumerate(zip(detections, padded_boxes, masks), start=1):
                box = clamp_box_xyxy(padded_box, width, height)
                point, point_source = _select_point(box, mask_record, prefer_mask_centroid)
                anchors.append(
                    {
                        "anchor_id": _make_anchor_id(str(sample["sample_id"]), img_idx, rank),
                        "img_idx": img_idx,
                        "name": det.get("phrase") or det.get("query"),
                        "query": det.get("query"),
                        "image_size": [width, height],
                        "bbox": box,
                        "bbox_1000": normalize_box_to_1000(box, width, height),
                        "point": point,
                        "point_1000": normalize_point_to_1000(point, width, height),
                        "score": float(det.get("score", 0.0)),
                        "mask_score": float(mask_record.get("score", 0.0)),
                        "point_source": point_source,
                        "source": {"detector": det.get("source"), "segmentor": mask_record.get("source")},
                    }
                )
        outputs.append({**sample, "anchors": anchors, "quality": {"num_anchors": len(anchors)}})

    save_json(output_path, outputs)
    return outputs
