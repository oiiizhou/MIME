# MIME Benchmark

### Face Is Not All You Need: MIME Benchmark for Incomplete Multimodal Emotion Recognition

**Official repository of the ACM Multimedia 2026 (ACM MM '26) paper**

[![Paper](https://img.shields.io/badge/Paper-ACM%20MM%202026-6f42c1)](https://doi.org/10.1145/3767308.3838643) [![DOI](https://img.shields.io/badge/DOI-10.1145%2F3767308.3838643-1f6feb)](https://doi.org/10.1145/3767308.3838643) [![Code License: CC BY 4.0](https://img.shields.io/badge/Code%20License-CC%20BY%204.0-2ea44f)](https://creativecommons.org/licenses/by/4.0/) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)

<!-- Uncomment the line below and replace the ID once the arXiv version is announced. -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/XXXX.XXXXX) -->

**MIME: A Chain-of-Emotion (CoE) benchmark for incomplete multimodal emotion recognition. It provides structured reasoning annotations to support robust and interpretable emotion understanding when facial, visual, or audio cues are partially unavailable.**

This repository contains the official introduction and code implementation for the paper "**Face Is Not All You Need: MIME Benchmark for Incomplete Multimodal Emotion Recognition**" (accepted to ACM Multimedia 2026).

<p align="center">
  <a href="https://yuxinokk.github.io/MIME/"><b>[HomePage]</b></a> |
  <a href="https://doi.org/10.1145/3767308.3838643"><b>[Paper]</b></a> |
  <a href="https://drive.google.com/drive/folders/1RgBzBK7UJMWL43hEPQzmZTJtSI7TkHhe?usp=sharing"><b>[Dataset]</b></a> |
  <a href="./license.pdf"><b>[License]</b></a> |
  <a href="./supplementary_material.pdf"><b>[Appendix]</b></a>
</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/oiiizhou/MIME/main/assets/fig_pipeline.jpg">
    <img src="assets/fig_pipeline.jpg" alt="MIME overview: seven incomplete-modality subsets, the CoE generation pipeline, and the evaluation suite" width="95%">
  </a>
</p>

<p align="center"><sub><b>Overview of MIME.</b> <b>Top:</b> the seven incomplete-modality subsets and the three-part Chain-of-Emotion (CoE) prompt. <b>Middle:</b> the dataset construction pipeline, where samples passing ground-truth quality filtering go through multimodal cue extraction and remained-cue verification before the structured CoE generation. <b>Bottom:</b> the evaluation suite, which aggregates Context Perception (CPS), Affective Inference (AIS) and Label Consistency (LCS) into a Holistic CoE Index, validated against human judgments.</sub></p>

---

## Contents

- [News](#news) · [About MIME](#about-mime) · [MIME at a Glance](#mime-at-a-glance) · [Quick Start](#quick-start)
- [Evaluation Protocol](#evaluation-protocol) · [Leaderboard](#leaderboard) · [Paper-to-Code Map](#paper-to-code-map)
- [Repository Structure](#repository-structure) · [Natural Degradation](#natural-degradation) · [Availability](#availability) · [Citation](#citation) · [License](#license)

## News
* **[2026.11]** MIME is accepted to **ACM Multimedia 2026 (ACM MM '26)**. Paper: [10.1145/3767308.3838643](https://doi.org/10.1145/3767308.3838643).
* **[2026.04]** The MIME benchmark repository is created for ACM Multimedia 2026. Mini data and code will be updated continuously.

## About MIME

Real-world multimodal emotion recognition frequently encounters incomplete or compromised observations, such as natural occlusion, side or back-facing heads, low-resolution capture, blurred facial regions, or missing audio. Existing datasets, however, largely assume full-modality availability and provide only emotion labels. This makes it difficult to study how models infer emotions when key cues are degraded or absent.

To bridge this gap, we introduce a dedicated **Chain-of-Emotion (CoE)** benchmark for incomplete multimodal emotion recognition.

**Key Features:**
- **Seven incomplete-modality subsets:** MIME includes one full-modality subset and six incomplete subsets that cover mild facial degradation, severe facial degradation, visual missingness, audio missingness, and dual-modality loss.
- **Natural plus controlled incompleteness:** The benchmark combines naturally challenging source videos with controlled degradation, allowing both realism and clear subset-based analysis.
- **Structured reasoning (CoE):** Each sample contains a three-part reasoning chain: Scene Understanding, Emotional Analysis, and Conclusion.
- **Rigorous evaluation:** The benchmark keeps seven emotion categories balanced while preserving uneven case sizes to reflect real scenario diversity.

## MIME at a Glance

| | |
|---|---|
| **Scale** | 2,000 video segments |
| **Source data** | CAER, DFEW, MAFW, FERV39k, and in-the-wild social media videos |
| **Emotion categories** | Happy, Sad, Neutral, Anger, Disgust, Fear, Surprise (balanced via resampling) |
| **Incompleteness** | 7 subsets; face degradation realized by InsightFace-tracked Gaussian blur at 3 progressive levels |
| **Reasoning annotation** | Three-part CoE: Scene Understanding → Emotional Analysis → Conclusion |
| **Human verification** | 173 annotated samples by 3 experts; Spearman 0.9415, Cohen's κ 0.8542, ICC 0.9628 against the MLLM judge |

**Subset definitions.** `✓` = the cue is available, `✗` = the cue is removed or degraded away.

| Subset | Audio | Face detail | Face structure | Visual scene | What it simulates |
|---|:---:|:---:|:---:|:---:|---|
| **FM** (Full Modality) | ✓ | ✓ | ✓ | ✓ | Ideal, complete observations |
| **FDM** (Facial Detail Missing) | ✓ | ✗ | ✓ | ✓ | Mild blur; fine facial detail weakened but coarse layout kept |
| **FSM** (Facial Structure Missing) | ✓ | ✗ | ✗ | ✓ | Severe blur; facial structure removed, forcing body/scene reliance |
| **VMM** (Visual Modality Missing) | ✓ | ✗ | ✗ | ✗ | Audio-only, e.g. phone calls or fully occluded cameras |
| **FDAM** (Facial Detail + Audio Missing) | ✗ | ✗ | ✓ | ✓ | Distant, muted CCTV; infer from posture and scene |
| **FSAM** (Facial Structure + Audio Missing) | ✗ | ✗ | ✗ | ✓ | Hardest setting; only body and scene remain |
| **AMM** (Audio Modality Missing) | ✗ | ✓ | ✓ | ✓ | Visual-only, e.g. muted video or broken microphone |

## Quick Start

```bash
git clone https://github.com/oiiizhou/MIME.git
cd MIME

conda create -n mime python=3.10 -y
conda activate mime
pip install openai tqdm scikit-learn opencv-python
```

Set your model endpoint once (the scripts read `DASHSCOPE_API_KEY` from the environment; an OpenAI-compatible base URL works as well):

```bash
export DASHSCOPE_API_KEY="your-key-here"
```

**1. Classification evaluation** — runs an MLLM over the clips, writes `predictions_<model>.jsonl`, and reports ACC / WF1 per subset:

```bash
python eval/predictcoe_evalacc.py --model gpt-5-mini
```

Use `--model all` to iterate over every model listed in `MODELS`.

**2. CoE quality evaluation** — scores the generated reasoning chains with an LLM-as-a-Judge and reports CPS / AIS / LCS / HCI:

```bash
python eval/evalcoe.py --pred_file predictions_gpt-5-mini.jsonl --workers 10
```

> **Note on reproducibility.** The released `data/` and `label.jsonl` contain only the 28-clip sample set (4 clips per subset). The numbers in the paper are computed on the full 2,000-segment test set, which is available after signing `license.pdf`. Running the commands above on the sample set will produce much noisier numbers.

## Evaluation Protocol

Beyond discrete accuracy, MIME scores the *reasoning process* through an LLM-as-a-Judge (default: `qwen3.5-plus`). Each criterion is scored in `[0, 10]` and averaged over the modalities that are actually valid in a given subset:

| Metric | Meaning |
|---|---|
| **CPS** — Context Perception Score | Whether the model correctly perceives the *surviving* physical cues (scene, body, audio). Computed over valid modalities only. |
| **AIS** — Affective Inference Score | Whether the emotional inference drawn from those cues is logically sound. Explicitly penalizes hallucinated cues. |
| **LCS** — Label Consistency Score | 10 if the CoE conclusion agrees with the final predicted label, otherwise 0. |
| **HCI** — Holistic CoE Index | `0.4 · CPS + 0.3 · AIS + 0.3 · LCS`, the headline measure of affective interpretability. |

The judge is validated against human annotation on 173 samples: Spearman **0.9415**, Cohen's κ **0.8542**, ICC **0.9628**.

## Leaderboard

**Overall performance (%) on the full MIME test set.** Best per column in **bold**. HCI uses α = 0.4, β = 0.3.

| Model | F1 | CPS | AIS | LCS | HCI |
|---|---:|---:|---:|---:|---:|
| Gemini-3.1-flash | **46.61** | 6.54 | 5.24 | **4.68** | 5.59 |
| Doubao-seed-2-0 | 45.27 | 6.43 | 4.96 | 4.65 | 5.46 |
| Qwen3.5-plus | 42.60 | **6.70** | **5.32** | 4.43 | **5.60** |
| Qwen3.5-27b | 40.99 | 6.57 | 5.13 | 4.32 | 5.46 |
| GPT-5-mini | 33.52 | 5.91 | 4.42 | 3.48 | 4.73 |
| GPT-5-nano | 27.01 | 5.53 | 4.06 | 2.95 | 4.31 |
| GPT-5.4-nano | 26.21 | 5.41 | 3.97 | 2.93 | 4.23 |

**Accuracy (%) per subset.** WF1 per subset is reported in the paper.

| Subset | Gemini-3.1-flash | Doubao-seed-2-0 | Qwen3.5-plus | Qwen3.5-27b | GPT-5-mini | GPT-5-nano | GPT-5.4-nano |
|---|---:|---:|---:|---:|---:|---:|---:|
| FM | 46.10 | **51.13** | 48.12 | 44.11 | 27.07 | 23.75 | 22.81 |
| FDM | **50.96** | 48.73 | 44.94 | 48.73 | 24.68 | 20.89 | 25.95 |
| FSM | **49.38** | 48.13 | 40.25 | 42.74 | 36.93 | 28.22 | 31.12 |
| VMM | 44.25 | 41.50 | 42.50 | 42.61 | **50.00** | 49.50 | 43.50 |
| FDAM | 37.98 | **40.87** | 35.58 | 35.10 | 23.08 | 19.23 | 20.67 |
| FSAM | 48.42 | 47.92 | **50.26** | 46.35 | 44.27 | 35.94 | 31.94 |
| AMM | **50.75** | 47.12 | 46.00 | 43.50 | 31.75 | 21.75 | 25.00 |

Two takeaways from the paper: no single model dominates across subsets, and **CPS consistently exceeds AIS** for every model — MLLMs can describe a scene far better than they can reason about the emotion in it.

## Paper-to-Code Map

| Paper component | Code |
|---|---|
| Three-part CoE output format (Scene Understanding / Emotional Analysis / Conclusion) | `eval/evalcoe.py` → `get_judge_system_prompt()` |
| Valid modalities per subset (Table: FM … AMM) | `eval/evalcoe.py` → `get_valid_mods()` |
| CPS over valid modalities, eq. (1) | `eval/evalcoe.py` → `calculate_metrics()` → `metric1_scene` |
| AIS over valid modalities, eq. (2) | `eval/evalcoe.py` → `calculate_metrics()` → `metric2_emotion` |
| LCS, eq. (3) | `eval/evalcoe.py` → `calculate_metrics()` → `metric3_class` |
| HCI weighting (α = 0.4, β = 0.3), eq. (4) | `eval/evalcoe.py` → `WEIGHTS` / `metric4_overall` |
| MLLM-as-a-Judge calls with retries | `eval/evalcoe.py` → `call_judge()` |
| Frame sampling for MLLM input (8 frames per clip) | `eval/predictcoe_evalacc.py` → `extract_frames_base64()`, `NUM_FRAMES` |
| MLLM inference and prediction recording | `eval/predictcoe_evalacc.py` → `generate_prediction()` |
| ACC / WF1 computation | `eval/predictcoe_evalacc.py` → `print_metrics()` |
| Evaluated model list | `eval/predictcoe_evalacc.py` → `MODELS` |

## Repository Structure
- `data/`: Seven subset folders containing the video files. Each subset corresponds to a distinct incomplete-observation scenario:
  - **`Subset1_FM` (Full Modality):** Intact audio-visual clips with complete observations.
  - **`Subset2_FDM` (Mild Facial Degradation):** Light facial-region blur that weakens subtle facial details while preserving coarse layout.
  - **`Subset3_FSM` (Severe Facial Degradation):** Strong facial-region blur that substantially removes usable facial structure, forcing reliance on body, scene, and audio cues.
  - **`Subset4_VMM` (Visual Modality Missing):** Audio-only setting with visual frames removed.
  - **`Subset5_FDAM` (Mild Facial Degradation + Audio Missing):** Light facial degradation combined with removed audio.
  - **`Subset6_FSAM` (Severe Facial Degradation + Audio Missing):** Strong facial degradation combined with removed audio, creating the most challenging visual-only residual setting.
  - **`Subset7_AMM` (Audio Modality Missing):** Visual-only setting with audio removed.
- `data_list.txt`: Index of released sample files and metadata.
- `eval/`: Evaluation and testing scripts.
  - `predictcoe_evalacc.py`: Generates predicted emotion labels and CoE outputs, then computes hard metrics such as accuracy across subsets.
  - `evalcoe.py`: Uses an LLM-as-a-Judge protocol to evaluate the quality of model-generated CoE reasoning.
- `label.jsonl`: Structured CoE annotations for released samples.
- `natural_degradation/`: Qualitative clips that illustrate naturally occurring degradation already present in source videos before any controlled corruption is applied. This directory is provided for qualitative inspection only; it is **not** an additional benchmark subset.
  - `foreground_or_object_occlusion/`: 108 clips with foreground people, scene objects, or structures partially blocking the target face or body.
  - `side_facing_or_turning_away/`: 59 clips with clear side-facing poses, head turning, or incomplete frontal visibility.
  - `low_light_or_low_visibility/`: 21 clips with dark lighting, poor visibility, or severe underexposure.
  - `hand_or_body_occlusion/`: 5 clips with hands, arms, or body parts covering important facial cues.
  - `natural_degradation.csv`: Per-clip index for the released videos, including the English folder assignment together with the original Chinese tags and notes.
- `assets/`: Figures used in this README.
- `README.md`: This file.
- `LICENSE`: Licensing terms for the code and the benchmark.
- `license.pdf`: License document to sign for full benchmark access.
- `supplementary_material.pdf`: Appendix and supplementary materials.

## Natural Degradation
To make the source-video realism more transparent, we additionally release a small qualitative collection in `natural_degradation/`. These clips are grouped by their **primary naturally occurring degradation pattern** observed in the original videos, before any benchmark-side corruption is added.

This release is intended to support readers who want concrete examples of the kinds of real-world imperfections already present in MIME source data. Some clips may exhibit multiple challenging factors at once, but each clip is placed into one folder according to its dominant visible characteristic.

## Availability

The benchmark can be accessed in two ways:

- **Sample set (no registration).** 4 videos per subset (28 videos in total) are released directly in `data/`, together with `label.jsonl`.
- **Full benchmark (2,000 segments).** Sign `license.pdf` and send the signed file to `jinj62062@gmail.com`, CC `lanx@cse.neu.edu.cn`.

## Citation

If you find MIME useful in your research, please cite our paper:

```bibtex
@inproceedings{jia2026mime,
  author    = {Jia, Yuxin and Lan, Xing and Wang, Wensong and Xue, Jian and Ren, Feiliang and Lu, Ke},
  title     = {Face Is Not All You Need: MIME Benchmark for Incomplete Multimodal Emotion Recognition},
  year      = {2026},
  isbn      = {9798400722134},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3767308.3838643},
  doi       = {10.1145/3767308.3838643},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  pages     = {16784--16791},
  numpages  = {8},
  keywords  = {incomplete multimodal emotion recognition, multimodal large language models, benchmark, chain of emotion, emotion recognition},
  location  = {Rio de Janeiro, Brazil},
  series    = {MM '26}
}
```

## License

- **Code** (everything under `eval/`) is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
- **Benchmark data** is released for **academic and non-commercial research only**. Access to the full 2,000-segment set requires a signed `license.pdf`; redistribution is not permitted.

See [`LICENSE`](./LICENSE) and [`license.pdf`](./license.pdf) for details.
