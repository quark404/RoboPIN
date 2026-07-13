import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "Qwen3VLForConditionalGeneration is required. Please install a recent "
        "Transformers version that supports Qwen3-VL."
    ) from exc


DEFAULT_MODEL_PATH = "QwQ2/RoboPIN-4B"


def parse_args():
    parser = argparse.ArgumentParser(description="Run RoboPIN-4B inference.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="Input image path. Repeat for multi-view inference.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    return parser.parse_args()


def dtype_from_name(name):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_images(paths):
    images = []
    for image_path in paths:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def format_instruction():
    return (
        "\nThe answer should be presented in JSON format. Provide your thinking "
        "process between the <think> and </think> tags, and then give your final "
        "answer between the <answer> and </answer> tags."
    )


def extract_json_blocks(text):
    blocks = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    parsed = []
    for block in blocks:
        try:
            parsed.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    return parsed


def main():
    args = parse_args()
    images = load_images(args.image)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": args.prompt + format_instruction()})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(output)
    parsed = extract_json_blocks(output)
    if parsed:
        print("\nParsed JSON:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
