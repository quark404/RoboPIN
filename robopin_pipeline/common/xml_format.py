import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image


def _clamp_1000(value: float) -> int:
    return max(0, min(1000, int(round(value))))


def normalize_point_to_1000(point: Sequence[float], width: int, height: int) -> List[int]:
    x, y = [float(v) for v in point[:2]]
    width = max(int(width), 1)
    height = max(int(height), 1)
    return [_clamp_1000((x * 1000.0) / width), _clamp_1000((y * 1000.0) / height)]


def normalize_box_to_1000(box: Sequence[float], width: int, height: int) -> List[int]:
    x1, y1 = normalize_point_to_1000(box[:2], width, height)
    x2, y2 = normalize_point_to_1000(box[2:4], width, height)
    return [x1, y1, x2, y2]


def serialize_answer(answer_raw: Any) -> str:
    if answer_raw is None:
        return ""
    if isinstance(answer_raw, str):
        return answer_raw
    return json.dumps(answer_raw, ensure_ascii=False)


def sample_image_paths(sample: Dict[str, Any]) -> List[str]:
    image_paths = sample.get("image")
    if isinstance(image_paths, str):
        return [image_paths]
    if isinstance(image_paths, list):
        return [str(path) for path in image_paths]

    paths: List[str] = []
    for rec in sample.get("images") or []:
        if isinstance(rec, dict) and rec.get("path"):
            paths.append(str(rec["path"]))
    return paths


def resolve_image_path(image_root: str, image_path: str) -> str:
    if not image_root:
        return image_path
    candidate = Path(image_path)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(image_root) / image_path)


def _read_image_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


def build_image_size_map(sample: Dict[str, Any], image_root: str = "") -> Dict[int, Tuple[int, int]]:
    size_map: Dict[int, Tuple[int, int]] = {}
    records = sample.get("images") or []
    if records:
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict) or not rec.get("path"):
                continue
            img_idx = int(rec.get("img_idx", idx))
            size_map[img_idx] = _read_image_size(resolve_image_path(image_root, str(rec["path"])))
        return size_map
    for idx, path in enumerate(sample_image_paths(sample)):
        size_map[idx] = _read_image_size(resolve_image_path(image_root, path))
    return size_map


def _escape_attr_text(text: str) -> str:
    return str(text or "").replace("\\", "\\\\").replace('"', '\\"')


def format_relative_point_tag(name: str, img_idx: int, point_1000: Sequence[int]) -> str:
    point_text = f"[{int(point_1000[0])}, {int(point_1000[1])}]"
    return f'<tag_name name="{_escape_attr_text(name)}" img_idx="{int(img_idx)}" point="{point_text}">'


def format_relative_box_tag(name: str, img_idx: int, box_1000: Sequence[int]) -> str:
    box_text = f"[{int(box_1000[0])}, {int(box_1000[1])}, {int(box_1000[2])}, {int(box_1000[3])}]"
    return f'<tag_box name="{_escape_attr_text(name)}" img_idx="{int(img_idx)}" box="{box_text}">'


def anchor_point_1000(anchor: Dict[str, Any], size_map: Dict[int, Tuple[int, int]]) -> Optional[List[int]]:
    point_1000 = anchor.get("point_1000")
    if isinstance(point_1000, list) and len(point_1000) >= 2:
        return [int(point_1000[0]), int(point_1000[1])]
    point = anchor.get("point")
    if not isinstance(point, list) or len(point) < 2:
        return None
    img_idx = int(anchor.get("img_idx", 0))
    if img_idx not in size_map:
        return None
    width, height = size_map[img_idx]
    return normalize_point_to_1000(point, width, height)


def anchor_box_1000(anchor: Dict[str, Any], size_map: Dict[int, Tuple[int, int]]) -> Optional[List[int]]:
    box_1000 = anchor.get("bbox_1000")
    if isinstance(box_1000, list) and len(box_1000) >= 4:
        return [int(v) for v in box_1000[:4]]
    box = anchor.get("bbox")
    if not isinstance(box, list) or len(box) < 4:
        return None
    img_idx = int(anchor.get("img_idx", 0))
    if img_idx not in size_map:
        return None
    width, height = size_map[img_idx]
    return normalize_box_to_1000(box, width, height)
