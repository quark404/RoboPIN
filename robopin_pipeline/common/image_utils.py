from typing import Any, List, Sequence, Tuple

from PIL import Image


def load_pil_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def clamp_box_xyxy(box: Sequence[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def pad_box_xyxy(
    box: Sequence[float],
    width: int,
    height: int,
    padding_ratio: float,
) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * float(padding_ratio)
    pad_y = box_height * float(padding_ratio)
    return clamp_box_xyxy([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)


def center_from_box(box: Sequence[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))


def centroid_from_mask(mask: Any) -> Tuple[int, int]:
    import numpy as np

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return 0, 0
    y = float(coords[:, 0].mean())
    x = float(coords[:, 1].mean())
    return int(round(x)), int(round(y))
