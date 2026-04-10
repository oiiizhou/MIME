# MIME Dataset

**MIME: The first Chain-of-Emotion (CoE) dataset for multimodal emotion recognition under missing modalities (e.g., blurred faces). It provides structured reasoning annotations to logically deduce emotions when key cues are absent, enabling robust and interpretable Emotion AI in the wild.**

This repository contains the official introduction and codes implementation for the paper "**MIME: Missing Information in Multimodal Emotion**" (submitted to ACM Multimedia 2026 Dataset Track). 

<p align="center">
  <a href="https://yuxinokk.github.io/MIME/"><b>[HomePage]</b></a> ⋅ 
  <a href="https://drive.google.com/drive/folders/1RgBzBK7UJMWL43hEPQzmZTJtSI7TkHhe?usp=sharing"><b>[Dataset]</b></a> ⋅
  <a href="./license.pdf"><b>[License]</b></a> ⋅
  <a href="./supplementary_material.pdf"><b>[Appendix]</b></a>
</p>

---

## 📰 News
* **[2026.04]** The MIME dataset repository is created for ACM Multimedia 2026. Mini data and codes will be fully updated by April 8, 2026.


## 📊 About MIME

Real-world multimodal emotion recognition frequently encounters the challenge of missing or compromised modalities, such as blurred faces due to privacy preservation or missing audio in restricted environments. However, existing datasets predominantly assume ideal, full-modality availability and provide only discrete emotion labels. This leaves a critical gap in understanding *how* to logically deduce emotions when key modalities are absent. 

To bridge this gap, we introduce the first **Chain-of-Emotion (CoE)** dataset tailored for emotion recognition under missing modalities. 

**Key Features:**
- **Diverse Scenarios:** Built upon authoritative video benchmarks like CAER and DFEW, we systematically categorize data into five scenarios: one full-modality baseline and four specific missing-modality settings.
- **Adaptive Data Construction Pipeline:** Leveraging large multimodal models with a conditional blurred-cue verification stage for face-degraded scenarios to prevent model hallucination.
- **Structured Reasoning (CoE):** Annotations follow a structured three-part format (Scene Understanding, Emotion Analysis, Conclusion) to explicitly capture how alternative cues compensate for missing information.
- **Rigorous Evaluation:** Covering seven balanced emotion categories. Note: the dataset now contains seven missing-modality cases with uneven sample counts across cases, while the seven emotion categories remain balanced.

## 📁 Repository Structure
- `data/`: Seven missing-modality case folders: `CASE1_FM`, `CASE2_FDM`, `CASE3_FSM`, `CASE4_VMM`, `CASE5_FDAM`, `CASE6_FSAM`, `CASE7_AMM`.
- `data_list.txt`: Index of released sample files and metadata.
- `eval/`: Evaluation and testing scripts.
  - `predictcoe_evalacc.py`: Script to invoke the model, generate emotion labels and Chain-of-Emotion (CoE) from audio/video inputs, and evaluate hard metrics (e.g., accuracy).
  - `eval_coe.py`: Script leveraging an LLM-as-a-Judge approach to evaluate the logical reasoning and quality of the model-generated CoE.
- `label.jsonl`: Structured CoE annotations for released samples.
- `README.md`: This file.
- `license.pdf`: License document for dataset access and usage.
- `supplementary_material.pdf`: Appendix and supplementary materials.

## 📈 Evaluation
We provide comprehensive tools in the `eval/` directory to assess both the classification performance and the reasoning capabilities of multimodal models:

* **Hard Metrics Evaluation (`predictcoe_evalacc.py`):** This script passes audio and video data to your model to obtain the predicted emotion categories alongside the generated Chain-of-Emotion. It automatically calculates quantitative hard metrics, such as classification accuracy (ACC), across the different missing-modality cases.
* **CoE Quality Evaluation (`eval_coe.py`):** To assess the interpretability of the model, this script uses a judge model to evaluate the generated Chain-of-Emotion. It systematically scores the model's output against the ground-truth reasoning structures to ensure logical consistency and correctness when dealing with absent cues.

## 📦 Availability
The benchmark can be accessed in two ways. To quickly view, you are welcome to directly download a small sample set containing 4 videos per case (28 videos in total). If you would like to use the full benchmark, kindly sign the `license.pdf` and send it to `jinj62062@gmail.com` (CC: `lanx@cse.neu.edu.cn`).

## 📝 License
The dataset and code in this repository are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. See `license.pdf` for the uploaded license document.
