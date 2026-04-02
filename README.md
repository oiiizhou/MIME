# MIME Dataset

**MIME: The first large-scale Chain-of-Emotion (CoE) dataset for multimodal emotion recognition under missing modalities (e.g., blurred faces). It provides structured reasoning annotations to logically deduce emotions when key cues are absent, enabling robust and interpretable Emotion AI in the wild.**

This repository contains the official introduction and codes implementation for the paper "**MIME: Missing Information in Multimodal Emotion**" (submitted to ACM Multimedia 2026 Dataset Track). 

<p align="center">
  <a href="#mime-dataset"><b>[Paper]</b></a> ⋅ 
  <a href="#mime-dataset"><b>[WebPage]</b></a> ⋅ 
  <a href="#mime-dataset"><b>[Dataset]</b></a>
</p>

---

## 📰 News
* **[2026.04]** The MIME dataset repository is created for ACM Multimedia 2026. Data and codes will be fully updated by April 8, 2026.

## 🛠️ Requirements
To be updated... (Detailed environment setup and dependencies will be released soon).

## 📊 About MIME

Real-world multimodal emotion recognition frequently encounters the challenge of missing or compromised modalities, such as blurred faces due to privacy preservation or missing audio in restricted environments. However, existing datasets predominantly assume ideal, full-modality availability and provide only discrete emotion labels. This leaves a critical gap in understanding *how* to logically deduce emotions when key modalities are absent. 

To bridge this gap, we introduce the first large-scale **Chain-of-Emotion (CoE)** dataset tailored for emotion recognition under missing modalities. 

**Key Features:**
- **Diverse Scenarios:** Built upon authoritative video benchmarks like CAER and DFEW, we systematically categorize data into five scenarios: one full-modality baseline and four specific missing-modality settings.
- **Adaptive Data Construction Pipeline:** Leveraging large multimodal models with a conditional blurred-cue verification stage for face-degraded scenarios to prevent model hallucination.
- **Structured Reasoning (CoE):** Annotations follow a structured three-part format (Scene Understanding, Emotion Analysis, Conclusion) to explicitly capture how alternative cues compensate for missing information.
- **Rigorous Evaluation:** Covering seven balanced emotion categories with a rigorous test set of 400 samples per scenario.

## 📁 Repository Structure
* `dataset/`: Contains the core dataset including `merged_data` and `merged.jsonl`.
* `scripts/`: Contains scripts for data processing across different missing-modality cases and CoE generation via LLMs.
* `eval/`: Contains testing inference codes, evaluation scripts, and experimental results.

## 📝 License
To be updated...
