# MIME Benchmark

**MIME: The first Chain-of-Emotion (CoE) benchmark for multimodal emotion recognition under missing modalities (e.g., blurred faces). It provides structured reasoning annotations to logically deduce emotions when key cues are absent, enabling robust and interpretable Emotion AI in the wild.**

This repository contains the official introduction and codes implementation for the paper "**MIME: Missing Information in Multimodal Emotion**" (submitted to ACM Multimedia 2026 Benchmark Track). 

<p align="center">
  <a href="https://yuxinokk.github.io/MIME/"><b>[HomePage]</b></a> ⋅ 
  <a href="https://drive.google.com/drive/folders/1RgBzBK7UJMWL43hEPQzmZTJtSI7TkHhe?usp=sharing"><b>[Dataset]</b></a> ⋅
  <a href="./license.pdf"><b>[License]</b></a> ⋅
  <a href="./supplementary_material.pdf"><b>[Appendix]</b></a>
</p>

---

## 📰 News
* **[2026.04]** The MIME benchmark repository is created for ACM Multimedia 2026. Mini data and codes will be fully updated by April 8, 2026.


## 📊 About MIME

Real-world multimodal emotion recognition frequently encounters the challenge of missing or compromised modalities, such as blurred faces due to privacy preservation or missing audio in restricted environments. However, existing datasets predominantly assume ideal, full-modality availability and provide only discrete emotion labels. This leaves a critical gap in understanding *how* to logically deduce emotions when key modalities are absent. 

To bridge this gap, we introduce the first **Chain-of-Emotion (CoE)** benchmark tailored for emotion recognition under missing modalities. 

**Key Features:**
- **Diverse Scenarios:** Built upon authoritative video benchmarks like CAER and DFEW, we systematically categorize data into five scenarios: one full-modality baseline and four specific missing-modality settings.
- **Adaptive Data Construction Pipeline:** Leveraging large multimodal models with a conditional blurred-cue verification stage for face-degraded scenarios to prevent model hallucination.
- **Structured Reasoning (CoE):** Annotations follow a structured three-part format (Scene Understanding, Emotion Analysis, Conclusion) to explicitly capture how alternative cues compensate for missing information.
- **Rigorous Evaluation:** Covering seven balanced emotion categories. Note: the benchmark now contains seven missing-modality subsets with uneven sample counts across subsets, while the seven emotion categories remain balanced.

## 📁 Repository Structure
- `data/`: Seven missing-modality subset folders containing the video files. Each subset systematically simulates a distinct real-world scenario of information loss:
  - **`CASE1_FM` (Full Modality):** Intact audio-visual clips providing complete information, representing ideal, controlled environments like professional broadcasts or high-quality recordings.
  - **`CASE2_FDM` (Face Details Missing):** Applies a light 2D Gaussian blur (k=15, σ=0) strictly within a precise facial mask. This simulates mild visual imperfections (e.g., low-resolution webcams, compression artifacts) where subtle micro-expressions are lost but the general facial layout remains perceptible.
  - **`CASE3_FSM` (Face Structures Missing):** Applies a progressively heavier Gaussian blur (k ∈ {35, 55}, σ=0) to severely degrade facial structures, rendering the face as an unrecognizable color blob. Replicates severe real-world occlusions (e.g., heavy masks, extreme lighting failures), forcing models to deduce emotions entirely from body language and scene context.
  - **`CASE4_VMM` (Visual Modality Missing):** Simulates an audio-only scenario where all visual frames are discarded (replaced by black frames). Mirrors everyday situations like phone calls, voice messages, or completely occluded cameras.
  - **`CASE5_FDAM` (Face Details and Audio Modality Missing):** A dual-modality loss condition combining the light facial blur from Case 2 (k=15) and removing the audio track. Mimics highly adverse environments like distant, low-resolution, and muted CCTV surveillance.
  - **`CASE6_FSAM` (Face Structures and Audio Modality Missing):** The most extreme visual degradation combined with audio loss. The face is heavily blurred (k ∈ {35, 55}) and audio is muted, rigorously testing the ability to infer emotions entirely from body posture and surrounding environmental dynamics.
  - **`CASE7_AMM` (Audio Modality Missing):** Simulates a visual-only scenario by removing the audio track. Corresponds to viewing muted videos, interacting through soundproof barriers, or encountering hardware microphone failures.
- `data_list.txt`: Index of released sample files and metadata.
- `eval/`: Evaluation and testing scripts.
  - `predictcoe_evalacc.py`: Script to invoke the model, generate emotion labels and Chain-of-Emotion (CoE) from audio/video inputs, and evaluate hard metrics (e.g., accuracy).
  - `eval_coe.py`: Script leveraging an LLM-as-a-Judge approach to evaluate the logical reasoning and quality of the model-generated CoE.
- `label.jsonl`: Structured CoE annotations for released samples.
- `README.md`: This file.
- `license.pdf`: License document for benchmark access and usage.
- `supplementary_material.pdf`: Appendix and supplementary materials.

## 📈 Evaluation
We provide comprehensive tools in the `eval/` directory to assess both the classification performance and the reasoning capabilities of multimodal models:

* **Hard Metrics Evaluation (`predictcoe_evalacc.py`):** This script passes audio and video data to your model to obtain the predicted emotion categories alongside the generated Chain-of-Emotion. It automatically calculates quantitative hard metrics, such as classification accuracy (ACC), across the different missing-modality subsets.
* **CoE Quality Evaluation (`eval_coe.py`):** To assess the interpretability of the model, this script uses a judge model to evaluate the generated Chain-of-Emotion. It systematically scores the model's output against the ground-truth reasoning structures to ensure logical consistency and correctness when dealing with absent cues.

## 📦 Availability
The benchmark can be accessed in two ways. To quickly view, you are welcome to directly download a small sample set containing 4 videos per subset (28 videos in total). If you would like to use the full benchmark, kindly sign the `license.pdf` and send it to `jinj62062@gmail.com` (CC: `lanx@cse.neu.edu.cn`).

## 📝 License
The benchmark and code in this repository are released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. See `license.pdf` for the uploaded license document.
