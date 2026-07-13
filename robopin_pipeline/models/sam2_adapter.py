import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image


class SAM2Adapter:
    """SAM 2 image predictor wrapper."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda")
        self.predictor = None
        self._load()

    def _load(self) -> None:
        repo_path = self.config.get("repo_path")
        config_path = self.config.get("config_path")
        checkpoint_path = self.config.get("checkpoint_path")
        if not repo_path or not config_path or not checkpoint_path:
            raise ValueError("SAM 2 config requires repo_path, config_path, and checkpoint_path")

        repo_root = Path(repo_path).resolve()
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        build_module = importlib.import_module("sam2.build_sam")
        predictor_module = importlib.import_module("sam2.sam2_image_predictor")
        build_sam2 = getattr(build_module, "build_sam2")
        predictor_cls = getattr(predictor_module, "SAM2ImagePredictor")

        model = build_sam2(self._normalize_config_path(repo_root, str(config_path)), checkpoint_path, device=self.device)
        self.predictor = predictor_cls(model)

    @staticmethod
    def _normalize_config_path(repo_root: Path, config_path: str) -> str:
        config_path = config_path.replace("\\", "/")
        if config_path.startswith("configs/"):
            return config_path
        if config_path.startswith("sam2/configs/"):
            return config_path[len("sam2/") :]
        candidate = Path(config_path)
        if not candidate.is_absolute():
            return config_path
        resolved = candidate.resolve()
        for root in (repo_root / "sam2", repo_root):
            try:
                rel_path = resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if rel_path.startswith("configs/"):
                return rel_path
        parts = list(resolved.parts)
        if "configs" in parts:
            return Path(*parts[parts.index("configs") :]).as_posix()
        return resolved.as_posix()

    def segment_from_boxes(
        self,
        image: Image.Image,
        boxes_xyxy: Sequence[Sequence[float]],
    ) -> List[Dict[str, Any]]:
        self.predictor.set_image(np.asarray(image.convert("RGB")))
        outputs: List[Dict[str, Any]] = []
        for box in boxes_xyxy:
            box_np = np.asarray(box, dtype=np.float32)[None, :]
            masks, scores, _ = self.predictor.predict(box=box_np, multimask_output=False)
            if masks is None or len(masks) == 0:
                outputs.append({"mask": None, "score": 0.0, "source": "sam2"})
                continue
            score = float(scores[0]) if scores is not None and len(scores) > 0 else 0.0
            outputs.append({"mask": masks[0].astype(np.uint8), "score": score, "source": "sam2"})
        return outputs
