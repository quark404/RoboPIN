import base64
import json
import random
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image


class LocalVLMAPI:
    """OpenAI-compatible local VLM client, tested with vLLM-served Qwen3-VL."""

    def __init__(
        self,
        api_key: str = "EMPTY",
        base_url: str = "",
        model_name: str = "",
        timeout: int = 600,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
        retry_jitter_sec: float = 1.0,
        **_: Any,
    ):
        if not base_url or not model_name:
            raise ValueError("LocalVLMAPI requires base_url and model_name")
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url, timeout=timeout)
        self.model_name = model_name
        self.max_retries = max(1, int(max_retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))
        self.retry_jitter_sec = max(0.0, float(retry_jitter_sec))

    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate(
        self,
        images: Sequence[Image.Image],
        prompt_text: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        content: List[Dict[str, Any]] = []
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(image)}"},
                }
            )
        content.append({"type": "text", "text": prompt_text})

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                output = (response.choices[0].message.content or "").strip()
                if not output:
                    raise RuntimeError("VLM returned empty content")
                return output
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_delay_sec + random.uniform(0.0, self.retry_jitter_sec))
        raise RuntimeError(f"VLM request failed after {self.max_retries} attempts: {last_error}") from last_error


def strip_code_fence(text: str) -> str:
    raw = (text or "").strip()
    if "```" not in raw:
        return raw
    parts = raw.split("```")
    if len(parts) >= 3:
        middle = parts[1].strip()
        lines = middle.splitlines()
        if lines and lines[0].strip().lower() in {"json", "text"}:
            middle = "\n".join(lines[1:]).strip()
        return middle
    return raw.replace("```", "").strip()


def parse_json_response(text: str) -> Any:
    cleaned = strip_code_fence(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        return None
