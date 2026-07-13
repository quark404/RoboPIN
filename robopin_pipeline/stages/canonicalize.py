import re
from typing import Any, Dict, List, Tuple

from robopin_pipeline.common.io import load_json, save_json


IMAGE_TOKEN_PATTERN = re.compile(r"(?:<image>\s*)+")


def _clean_text(value: Any) -> str:
    return IMAGE_TOKEN_PATTERN.sub("", str(value or "")).strip()


def _extract_question_and_answer(row: Dict[str, Any]) -> Tuple[str, Any]:
    question = row.get("question") or row.get("problem") or row.get("prompt") or ""
    answer = row.get("answer")
    conversations = row.get("conversations")
    if isinstance(conversations, list) and conversations:
        if not question and isinstance(conversations[0], dict):
            question = _clean_text(conversations[0].get("value", ""))
        if answer is None and len(conversations) > 1 and isinstance(conversations[1], dict):
            answer = _clean_text(conversations[1].get("value", ""))
    return _clean_text(question), answer


def _extract_images(row: Dict[str, Any]) -> List[str]:
    raw_images = row.get("images")
    if raw_images is None:
        raw_images = row.get("image")
    if raw_images is None:
        raw_images = row.get("image_paths")
    if isinstance(raw_images, str):
        return [raw_images]
    if isinstance(raw_images, list):
        return [str(path) for path in raw_images]
    return []


def _infer_answer_type(answer: Any) -> str:
    if isinstance(answer, dict):
        return "json"
    text = str(answer or "").strip()
    if '"point_2d"' in text or "point_2d" in text:
        return "point_or_json"
    if re.fullmatch(r"[A-Z]", text):
        return "choice"
    return "text"


def canonicalize_rows(rows: List[Dict[str, Any]], dataset_name: str) -> List[Dict[str, Any]]:
    canonical: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question, answer = _extract_question_and_answer(row)
        images = _extract_images(row)
        sample_id = str(row.get("id") or row.get("sample_id") or f"{dataset_name}_{idx:08d}")
        canonical.append(
            {
                "sample_id": sample_id,
                "dataset": dataset_name,
                "task_family": row.get("task_family") or row.get("prompt_family") or "unknown",
                "question": question,
                "answer_raw": answer,
                "answer_type": row.get("answer_type") or _infer_answer_type(answer),
                "images": [{"img_idx": i, "path": path} for i, path in enumerate(images)],
                "raw_meta": dict(row),
            }
        )
    return canonical


def run_canonicalize(input_path: str, output_path: str, dataset_name: str) -> List[Dict[str, Any]]:
    rows = load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError("Canonicalize expects a top-level JSON list")
    output = canonicalize_rows(rows, dataset_name)
    save_json(output_path, output)
    return output
