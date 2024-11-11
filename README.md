![mmoe](assets/mmoe.png)

<h1 align="center">MMoE: Enhancing Multimodal Models with Mixtures of Multimodal Interaction Experts</h1>

<div align="center">

  

[![Python 3.8](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/downloads/release/python-3109/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-orange)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![Arxiv](https://img.shields.io/badge/arXiv-2311.09580-b31b1b)](https://github.com/ulab-uiuc/research-town)

</div>

### Introduction

MMoE is an initial trial to build better multimodal foundation models with the help of the concept of multimodal interaction. In this work, multimodal interaction relationship between image and text modalities are split into three different types: redundnacy (when two modalities carry similar task-related information), uniqueness (when two modalities carry unique task-related information), and synergy (when two modalities interact with each other to synergize new task-related information). We build and train expert models for each interaction type and train a fuser to fuse them together as the final prediction.

### Get started

The overall pipeline of MMoE includes three steps:

1. Categorizing training data based on multimodal interactions
2. Training expert models for each multimodal interaction type
3. Inference with mixtures of expert models

In the following section, we provide detailed instruction on running experiments for each part.

### Categorization

To categorize multimodal datasets into interaction types (Redundancy, Uniqueness, Synergy), we use unimodal predictions from the **CogVLM2-LLaMA3-chat19B** for images and **Qwen2-72B-Instruct** for text. Below is an example using the MUSTARD dataset.

1. Step-by-Step Data Categorization

```bash
cd ./data_gen_vision_label_CogVLM2
pip install -r requirements.txt
python mustard_vision_label.py // collect unimodal vision label

cd ./data_gen_text_label_Qwen2
pip install -r requirements.txt
python mustard_text_label.py // collect unimodal text label

cd ./data_split
pip install -r requirements.txt
python mustard_split.py
```

2. Accessing Preprocessed Data

We provide RUS-categorized datasets, including data for the `MMSD`, `MUSTARD`, and `URFUNNY` datasets. Due to size limitations, the image data is hosted externally. To use these datasets:

- Download images from the provided Google Drive links
- Place them in the respective image folders (`/mmsd_data`, `/mustard_data`, `/urfunny_data`).

### Training

Train expert models and the fusion model based on the multimodal interaction type.

Example: Training BLIP-2-Based Expert and Fusion Models

To train BLIP-2 models for the MUSTARD dataset:

```
cd ./expert_BLIP2
pip install -r requirements.txt
sh blip2_mustard_train.sh
sh blip2_mustard_test.sh // generate prediction for each model for each datapoint

cd ./expert_fusion
sh mustard_fusion_train.sh
sh mustard_fusion_test.sh // generate rus logits for each datapoint
```

### Inference

Once models are trained, fusion-based inference combines expert predictions for final results.

```bash
cd ./expert_fusion
python fusion.py
```

### Citation

If you find MMoE useful in your research, please consider citing our paper:

```latex
@inproceedings{yu-etal-2024-mmoe,
    title = "{MMoE}: Enhancing Multimodal Models with Mixtures of Multimodal Interaction Experts",
    author = "Yu, Haofei and Qi, Zhengyang and Jang, Lawrence Keunho and Salakhutdinov, Russ and Morency, Louis-Philippe and Liang, Paul Pu",
    editor = "Al-Onaizan, Yaser and Bansal, Mohit and Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.558",
    pages = "10006--10030",
}

```
