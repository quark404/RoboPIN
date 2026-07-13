# RoboPIN

RoboPIN is a 4B embodied reasoning model trained with Pinned Chain-of-Thought (PinCoT), a structured reasoning paradigm that pins each reasoning step to visual evidence through reasoning anchors. These anchors bind task-relevant entities to names, identities, view indices, and spatial grounding so that reasoning remains traceable across steps and views.

Paper: **RoboPIN: Grounded Embodied Reasoning via Pinned Chain-of-Thought**

## Resources

- Model weights: [QwQ2/RoboPIN-4B](https://huggingface.co/QwQ2/RoboPIN-4B)
- Evaluation toolkit: [EmbodiedEvalKit](https://github.com/pickxiguapi/EmbodiedEvalKit)
- Evaluation project page: [embodied-r.github.io](https://embodied-r.github.io/)
- Data construction pipeline: [docs_data_pipeline.md](docs_data_pipeline.md)

## Highlights

- PinCoT links reasoning steps to explicit visual anchors instead of relying on implicit text-only references.
- RoboPIN is trained with three-stage post-training for embodied knowledge, structured reasoning, and process-supervised alignment.
- The 4B model is designed for embodied spatial reasoning, multi-view reasoning, and pointing tasks.

## Installation

```bash
cd RoboPIN
pip install -r requirements.txt
```

The model weights can be loaded directly from Hugging Face:

```bash
export MODEL_PATH=QwQ2/RoboPIN-4B
```

If Hugging Face access is unstable in your environment, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Quick Inference

Run single-image inference:

```bash
python examples/infer.py \
  --model-path QwQ2/RoboPIN-4B \
  --image /path/to/image.png \
  --prompt "Find the target object and answer with grounded reasoning."
```

Run multi-view inference:

```bash
python examples/infer.py \
  --model-path QwQ2/RoboPIN-4B \
  --image /path/to/view_0.png \
  --image /path/to/view_1.png \
  --prompt "Answer the task using both views and keep object identities consistent."
```

## Evaluation

We evaluate RoboPIN-4B using [EmbodiedEvalKit](https://github.com/pickxiguapi/EmbodiedEvalKit), a unified evaluation framework for embodied AI benchmarks.

Clone and install the evaluation toolkit:

```bash
git clone https://github.com/pickxiguapi/EmbodiedEvalKit.git
cd EmbodiedEvalKit
pip install -r requirements.txt
```

Then run a Qwen3-VL style benchmark command with RoboPIN weights:

```bash
python eval_erqa.py \
  --model_name RoboPIN-4B \
  --model_path QwQ2/RoboPIN-4B \
  --backbone qwen3 \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.8 \
  --tensor_parallel_size 2
```

For convenience, this repository includes `scripts/evaluate_with_embodiedevalkit.sh`, which wraps the same pattern.

## Data Construction Pipeline

This repository includes a lightweight public pipeline for constructing PinCoT-style data:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  semantic-parse \
  --input work/canonical.json \
  --output work/semantic.json
```

See [docs_data_pipeline.md](docs_data_pipeline.md) for the full `canonicalize -> semantic-parse -> grounding -> xml -> thinking -> filter` workflow.

## Citation

```bibtex
@misc{robopin2026,
  title  = {RoboPIN: Grounded Embodied Reasoning via Pinned Chain-of-Thought},
  author = {Anonymous},
  year   = {2026}
}
```

## License

This repository is released under the MIT License. Model weights and datasets may have additional terms; please check the corresponding Hugging Face pages before use.
