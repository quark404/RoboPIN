from typing import Any, Dict

from robopin_pipeline.common.io import load_json


def load_config(path: str) -> Dict[str, Any]:
    config = load_json(path)
    if not isinstance(config, dict):
        raise ValueError("Pipeline config must be a JSON object")
    return config
