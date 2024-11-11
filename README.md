![mmoe](assets/mmoe.png)

<h1 align="center">MMoE: Enhancing Multimodal Models with Mixtures of Multimodal Interaction Experts</h1>

<div align="center">



[![Python 3.8](https://img.shields.io/badge/python-%E2%89%A53.10-blue)](https://www.python.org/downloads/release/python-3109/)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-orange)](https://github.com/hiyouga/LLaMA-Factory/pulls)
[![Arxiv](https://img.shields.io/badge/arXiv-2311.09580-b31b1b)](https://arxiv.org/abs/2311.09580)

</div>

# Introduction

MMoE is an initial trial to create better multimodal foundation models by focusing on how different types of interactions occur between image and text data. In this work, we categorize interactions between these two modalities into three types:

- **Redundancy**: When both image and text contain similar information relevant to the task.
- **Uniqueness**: When each modality (image or text) holds different information that’s valuable for the task.
- **Synergy**: When the image and text combine in a way that produces new, useful information for the task.

We design and train expert models to handle each interaction type separately, and then a final model (a “fuser”) combines their outputs to make the final prediction.

# Get started

The overall pipeline of MMoE includes three steps:

1. Categorizing training data based on multimodal interactions
2. Training expert models for each multimodal interaction type
3. Inference with mixtures of expert models

In the following section, we provide detailed instruction on running experiments for each part.

# Categorization

To categorize multimodal datasets into interaction types (Redundancy, Uniqueness, Synergy), we use unimodal predictions from the **CogVLM2-LLaMA3-chat19B** for images and **Qwen2-72B-Instruct** for text. Below is an example using the MUSTARD dataset.

**Step-by-Step Data Categorization**

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

**Accessing Preprocessed Data**

We provide organized and processed data for the `MMSD2.0`, `MUSTARD`, and `URFUNNY` datasets. Each dataset contains the following components:

- **data_raw**: Original dataset files (note: large images are stored separately; see instructions below).
- **data_gen_output**: Unimodal labels and captions.
- **data_split_output**: Data split according to unimodal labels.
- **expert_inference_output**: Expert model outputs.

You can download each dataset at the following links:

- [MMSD2.0 Data](https://drive.google.com/file/d/15PNO7Ve4k0S2SvASs_3lOCDulzlosVKC/view?usp=share_link)
- [MUSTARD Data](https://drive.google.com/file/d/15PNO7Ve4k0S2SvASs_3lOCDulzlosVKC/view?usp=share_link)
- [URFUNNY Data](https://drive.google.com/file/d/1kY44ewjhC5eUpN_Bw-3GjOmK8d2W4d2Y/view?usp=share_link)

**Note on Large Images**: Each dataset’s `/data_raw` folder contains an `/images` directory, which includes files that are too large to store directly. If you wish to run experiments, please download these images and place them under `/data_raw/images` as indicated below:

- [MMSD2.0 Images](https://drive.google.com/file/d/1b6WAOqYKuYybDmaEyyVanN9Ffx8QdJsN/view?usp=share_link)
- [MUSTARD Images](https://drive.google.com/file/d/1z4kCFM4gO0o18hKpFLIVUnzLtlIJDc1m/view?usp=share_link)
- [URFUNNY Images](https://drive.google.com/file/d/1p_z3s1zyga9EoGdTcne8qlWR2zMsjdPE/view?usp=share_link)

Due to size limitations, the image data is hosted externally. To use these datasets:

- Download images from the provided Google Drive links
- Place them in the respective image folders (`/mmsd2.0_data`, `/mustard_data`, `/urfunny_data`).

# Training

Train expert models and the fusion model based on the multimodal interaction type.

Example: Training BLIP-2-Based Expert and Fusion Models

To train BLIP-2 models for the MUSTARD dataset:

```bash
cd ./expert_BLIP2
pip install -r requirements.txt
sh blip2_mustard_train.sh
sh blip2_mustard_test.sh // generate prediction for each model for each datapoint

cd ./expert_fusion
sh mustard_fusion_train.sh
sh mustard_fusion_test.sh // generate rus logits for each datapoint
```

# Inference

Once models are trained, fusion-based inference combines expert predictions for final results.

```bash
cd ./expert_fusion
python fusion.py
```

# Citation

If you find MMoE useful in your research, please consider citing our paper:

```bibtex
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
