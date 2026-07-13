import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

from PIL import Image

from robopin_pipeline.common.xml_format import normalize_box_to_1000


LOC_TOKEN_PATTERN = re.compile(r"<loc_\d+>")


def _choose_device(requested: str, torch_module: Any) -> str:
    if str(requested).startswith("cuda") and torch_module.cuda.is_available():
        return str(requested)
    return "cpu"


def _strip_location_tokens(text: Any) -> str:
    cleaned = LOC_TOKEN_PATTERN.sub("", str(text or "")).strip()
    return re.sub(r"\s+", " ", cleaned)


def _format_box_loc_tokens(box_1000: Sequence[int]) -> str:
    return "".join(f"<loc_{int(value)}>" for value in box_1000[:4])


class Florence2Adapter:
    """Florence-2 wrapper for open-vocabulary detection and region inspection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cpu"
        self.model = None
        self.processor = None
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, modeling_utils
        from transformers.dynamic_module_utils import get_imports

        model_path = self.config.get("model_path")
        if not model_path:
            raise ValueError("Florence-2 config requires model_path")

        model_root = Path(str(model_path)).resolve()
        self.device = _choose_device(str(self.config.get("device", "cuda")), torch)
        torch_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        def fixed_get_imports(filename: str) -> List[str]:
            imports = get_imports(filename)
            if str(filename).endswith("modeling_florence2.py") and "flash_attn" in imports:
                imports.remove("flash_attn")
            return imports

        def load_model() -> Any:
            return AutoModelForCausalLM.from_pretrained(
                str(model_root),
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation="eager",
            )

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            try:
                self.model = load_model().eval().to(self.device)
            except AttributeError as exc:
                if "_supports_sdpa" not in str(exc):
                    raise
                modeling_utils.PreTrainedModel._supports_sdpa = False
                self.model = load_model().eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(str(model_root), trust_remote_code=True)

    @staticmethod
    def _normalize_queries(text_queries: Sequence[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for query in text_queries:
            text = str(query or "").strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(text)
        return normalized

    def _run_task(
        self,
        image: Image.Image,
        task_prompt: str,
        text_input: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> Any:
        import torch

        prompt = task_prompt if text_input is None else task_prompt + str(text_input)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        torch_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=int(max_new_tokens),
                num_beams=3,
                do_sample=False,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        processed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=image.size,
        )
        if isinstance(processed, dict) and task_prompt in processed:
            return processed[task_prompt]
        return processed

    def detect(
        self,
        image: Image.Image,
        text_queries: Sequence[str],
        max_new_tokens: int = 256,
        max_boxes: int = 8,
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        for query in self._normalize_queries(text_queries):
            task_output = self._run_task(
                image=image,
                task_prompt="<OPEN_VOCABULARY_DETECTION>",
                text_input=query,
                max_new_tokens=max_new_tokens,
            )
            raw_boxes = task_output.get("bboxes") or task_output.get("boxes") or []
            raw_labels = (
                task_output.get("bboxes_labels")
                or task_output.get("boxes_labels")
                or task_output.get("labels")
                or []
            )
            labels = list(raw_labels) if isinstance(raw_labels, list) else []
            for idx, raw_box in enumerate(raw_boxes[: max(0, int(max_boxes))]):
                label = str(labels[idx]) if idx < len(labels) else query
                detections.append(
                    {
                        "phrase": label,
                        "query": query,
                        "bbox": [float(v) for v in raw_box],
                        "score": max(0.0, 1.0 - idx * 0.01),
                        "source": "florence2",
                    }
                )
        return detections

    def inspect_region(
        self,
        image: Image.Image,
        box_xyxy: Sequence[float],
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        width, height = image.size
        x1 = max(0, min(int(round(float(box_xyxy[0]))), width - 1))
        y1 = max(0, min(int(round(float(box_xyxy[1]))), height - 1))
        x2 = max(x1 + 1, min(int(round(float(box_xyxy[2]))), width))
        y2 = max(y1 + 1, min(int(round(float(box_xyxy[3]))), height))
        box = [x1, y1, x2, y2]
        loc_tokens = _format_box_loc_tokens(normalize_box_to_1000(box, width, height))

        return {
            "box": box,
            "region_category": _strip_location_tokens(
                self._run_task(image, "<REGION_TO_CATEGORY>", loc_tokens, max_new_tokens)
            ),
            "region_description": _strip_location_tokens(
                self._run_task(image, "<REGION_TO_DESCRIPTION>", loc_tokens, max_new_tokens)
            ),
            "region_ocr": _strip_location_tokens(
                self._run_task(image, "<REGION_TO_OCR>", loc_tokens, max_new_tokens)
            ),
        }
