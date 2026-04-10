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
* **[2026.04]** The MIME dataset repository is created for ACM Multimedia 2026. Data and codes will be fully updated by April 8, 2026.


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
- `eval/`: Evaluation and testing scripts (CoE evaluation tools and helpers).
- `label.jsonl`: Structured CoE annotations for released samples.
- `README.md`: This file.
- `license.pdf`: License document for dataset access and usage.
- `supplementary_material.pdf`: Appendix and supplementary materials.

## 📦 Availability
The benchmark can be accessed in two ways. To quickly view, you are welcome to directly download a small sample set containing 4 videos per case (28 videos in total). If you would like to use the full benchmark, kindly sign the `license.pdf` and send it to `jinj62062@gmail.com` (CC: `lanx@cse.neu.edu.cn`).

## 📝 License
The dataset and code in this repository are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. See `license.pdf` for the uploaded license document.
