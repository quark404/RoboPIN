# RoboPIN Data Construction Pipeline

This repository includes a public, lightweight version of the data construction pipeline used to create PinCoT-style training samples.

The implementation follows the paper logic:

1. **Semantic parsing**: a VLM parses the raw question into task-relevant entities, regions, and grounding queries.
2. **Grounding**: Florence-2 proposes candidate boxes for each query; SAM 2.1 refines masks; the pipeline extracts normalized point anchors and boxes.
3. **PinCoT generation**: a VLM receives the question, final answer, and candidate anchors, then writes a structured trace using identity-bound `<obj ...>` and `<space ...>` anchors.
4. **Filtering**: rule checks remove traces with invalid outer format, no anchors, duplicated first-introduction IDs, answer leakage, or correction/hesitation phrases.

## Input Format

The canonicalizer accepts a JSON list. Each row can use common fields:

```json
{
  "id": "sample_0001",
  "image": ["image0.png", "image1.png"],
  "question": "Where is the cup with respect to the plate?",
  "answer": "left"
}
```

It also supports ShareGPT-style `conversations`, `problem`, `images`, and `image_paths`.

## Commands

Canonicalize raw records:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  canonicalize \
  --input raw.json \
  --output work/canonical.json \
  --dataset-name my_dataset
```

Run semantic parsing:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  semantic-parse \
  --input work/canonical.json \
  --output work/semantic.json
```

Run Florence-2 + SAM 2 grounding:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  grounding \
  --input work/semantic.json \
  --output work/grounding.json
```

Convert anchors to XML-style candidate tags:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  xml \
  --input work/grounding.json \
  --output work/xml.json
```

Generate PinCoT traces:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  thinking \
  --input work/xml.json \
  --output work/thinking.json
```

Filter generated traces:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  filter \
  --input work/thinking.json \
  --output work/filtered.json
```

## Prompt Notes

The original internal pipeline used many dataset-specific semantic parsing prompts. The public release keeps a general prompt at `prompts/parse/generic_semantic_parse_prompt.txt`. It is designed to cover single-image spatial reasoning, pointing, affordance/contact-point, navigation, planning, temporal, and multi-view tasks.

The PinCoT generation prompt is at `prompts/thinking/pincot_generation_prompt.txt`. Compared with the appendix template, it explicitly requires the final `<answer>...</answer>` block and adds stronger anti-leakage constraints.

For production-scale generation, dataset-specific parse prompts are still recommended when a benchmark has unusual markers, options, or camera-motion wording.
