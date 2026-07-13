# RoboPIN: Grounded Embodied Reasoning via Pinned Chain-of-Thought (ACM MM26)

<div align="center">

**RoboPIN: Grounded Embodied Reasoning via Pinned Chain-of-Thought**

[[📄 Paper](https://arxiv.org/abs/2606.15753)] [[🤗 Models](https://huggingface.co/QwQ2/RoboPIN-4B)] [[🎯 Datasets](https://huggingface.co/datasets/QwQ2/RoboPIN-Datasets)] [[📊 Evaluation](https://github.com/pickxiguapi/EmbodiedEvalKit)] [[🧩 Data Pipeline](robopin_pipeline/docs_data_pipeline.md)] [[💬 Demo](#demo)]

</div>

---

## 🔥 Updates

- **[2026-07-13]** 🎉 Model weights released on Hugging Face: [QwQ2/RoboPIN-4B](https://huggingface.co/QwQ2/RoboPIN-4B).
- **[2026-07-13]** 🎯 Dataset repository initialized: [QwQ2/RoboPIN-Datasets](https://huggingface.co/datasets/QwQ2/RoboPIN-Datasets). Dataset files will be uploaded separately.
- **[2026-07-13]** 🧩 Public data construction pipeline released, including semantic parsing, Florence-2 + SAM 2 grounding, PinCoT generation prompts, and filtering utilities.

---

## 📖 Overview

**RoboPIN** is a 4B vision-language model for grounded embodied reasoning. It introduces **Pinned Chain-of-Thought (PinCoT)**, a structured reasoning paradigm that pins each reasoning step to visual evidence through identity-bound reasoning anchors.

Unlike text-only or coordinate-only chain-of-thought, PinCoT represents task-relevant objects and spaces with explicit anchors:

```text
<obj name="cup" id="obj_01" img_idx="0" point="[312, 476]">
<space name="middle plate" id="space_01" img_idx="1" point="[545, 623]">
```

These anchors bind semantic name, identity, view index, and spatial grounding, enabling the model to maintain consistent references across multi-step and multi-view reasoning.

RoboPIN is trained with a three-stage post-training recipe:

- **SFT** for embodied domain adaptation.
- **CoT-SFT** for PinCoT structured reasoning.
- **RFT** for process-supervised alignment over format, identity consistency, anchor localization, and final-answer correctness.

---

## 🛠️ Setup

1. **Clone the repository**

```bash
git clone https://github.com/quark404/RoboPIN.git
cd RoboPIN
```

2. **Create and activate an environment**

```bash
conda create -n robopin python=3.11 -y
conda activate robopin
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

If Hugging Face access is unstable in your environment, set:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 🚀 Inference

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

### 💬 Demo

Demo assets and additional examples will be added later. The current release provides the inference entrypoint and model weights.

---

## 🧩 Data Construction Pipeline

We provide a lightweight public implementation of the PinCoT data construction pipeline:

```text
semantic parsing -> grounding -> XML anchor export -> PinCoT generation -> filtering
```

Run semantic parsing:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  semantic-parse \
  --input work/canonical.json \
  --output work/semantic.json
```

Run the complete staged workflow:

```bash
python -m robopin_pipeline.runners.run_pipeline \
  --config configs/pipeline.example.json \
  full \
  --input work/canonical.json \
  --semantic-output work/semantic.json \
  --grounding-output work/grounding.json \
  --xml-output work/xml.json \
  --thinking-output work/thinking.json \
  --filtered-output work/filtered.json
```

See [robopin_pipeline/docs_data_pipeline.md](robopin_pipeline/docs_data_pipeline.md) for input formats, stage definitions, prompts, and filtering rules.

---

## 📊 Evaluation

We evaluate RoboPIN-4B using [EmbodiedEvalKit](https://github.com/pickxiguapi/EmbodiedEvalKit), a unified embodied AI evaluation framework.

Clone and install the toolkit:

```bash
git clone https://github.com/pickxiguapi/EmbodiedEvalKit.git
cd EmbodiedEvalKit
pip install -r requirements.txt
```

Run a Qwen3-VL style benchmark command with RoboPIN weights:

```bash
python eval_erqa.py \
  --model_name RoboPIN-4B \
  --model_path QwQ2/RoboPIN-4B \
  --backbone qwen3 \
  --max_model_len 20000 \
  --gpu_memory_utilization 0.8 \
  --tensor_parallel_size 2
```

This repository also includes a wrapper:

```bash
MODEL_PATH=QwQ2/RoboPIN-4B \
BENCHMARK_SCRIPT=eval_erqa.py \
bash scripts/evaluate_with_embodiedevalkit.sh
```

---

## 🎯 Datasets

The dataset repository has been initialized at:

[QwQ2/RoboPIN-Datasets](https://huggingface.co/datasets/QwQ2/RoboPIN-Datasets)

Dataset files and detailed data cards will be uploaded separately.

---

## 📜 Citation

If you use RoboPIN in your research, please cite our paper:

```bibtex
@article{huang2026robopin,
  title={RoboPIN: Grounded Embodied Reasoning via Pinned Chain-of-Thought},
  author={Huang, Yaoting and Yuan, Yifu and Han, Linqi and Li, Chengwen and Zhang, Shuoheng and Yao, Xianze and Tang, Hongyao and Zheng, Yan and Hao, Jianye},
  journal={arXiv preprint arXiv:2606.15753},
  year={2026}
}
```

## ⚖️ License

This repository is released under the MIT License. Model weights and datasets may have additional terms; please check the corresponding Hugging Face pages before use.
