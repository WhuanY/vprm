---
license: mit
annotations_creators:
  - expert-generated
  - found
language_creators:
  - expert-generated
  - found
task_categories:
- question-answering
- multiple-choice
- visual-question-answering
- text-generation
language:
- en
tags:
  - mathematics
  - reasoning
  - multi-modal-qa
  - math-qa
  - figure-qa
  - geometry-qa
  - math-word-problem
  - textbook-qa
  - vqa
  - geometry-diagram
  - synthetic-scene
  - chart
  - plot
  - scientific-figure
  - table
  - function-plot
  - abstract-scene
  - puzzle-test
  - document-image
  - science
configs:
  - config_name: default
    data_files:
      - split: test
        path: data/test-*
      - split: testmini
        path: data/testmini-*
pretty_name: MATH-V
size_categories:
- 1K<n<10K
---
# Measuring Multimodal Mathematical Reasoning with the MATH-Vision Dataset

[[💻 Github](https://github.com/mathllm/MATH-V/)] [[🌐 Homepage](https://mathllm.github.io/mathvision/)]  [[📊 Leaderboard ](https://mathllm.github.io/mathvision/#leaderboard)] [[📊 Open Source Leaderboard ](https://mathllm.github.io/mathvision/#openleaderboard)] [[🔍 Visualization](https://mathllm.github.io/mathvision/#visualization)] [[📖 Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/ad0edc7d5fa1a783f063646968b7315b-Paper-Datasets_and_Benchmarks_Track.pdf)]

## 🚀 Data Usage



<!-- **We have observed that some studies have used our MATH-Vision dataset as a training set.**
⚠️ **As clearly stated in our paper: *"The MATH-V dataset is not supposed, though the risk exists, to be used to train models for cheating. We intend for researchers to use this dataset to better evaluate LMMs’ mathematical reasoning capabilities and consequently facilitate future studies in this area."***

⚠️⚠️⚠️ **In the very rare situation that there is a compelling reason to include MATH-V in your training set, we strongly urge that the ***testmini*** subset be excluded from the training process!**
 -->

```python
from datasets import load_dataset

dataset = load_dataset("MathLLMs/MathVision")
print(dataset)
```

## 💥 News
- **[2025.05.16]** 💥 We now support the official open-source leaderboard! 🔥🔥🔥 [**Skywork-R1V2-38B**](https://github.com/SkyworkAI/Skywork-R1V) is the best open-source model, scoring **49.7%** on MATH-Vision. 🔥🔥🔥 [**MathCoder-VL-2B**](https://huggingface.co/MathLLMs/MathCoder-VL-2B) is the best small model on MATH-Vision, scoring **21.7%**. See the full [open-source leaderboard](https://mathllm.github.io/mathvision/#openleaderboard).
- **[2025.05.16]** 🤗 [MathCoder-VL-2B](https://huggingface.co/MathLLMs/MathCoder-VL-2B), [MathCoder-VL-8B](https://huggingface.co/MathLLMs/MathCoder-VL-8B) and [FigCodifier-8B](https://huggingface.co/MathLLMs/FigCodifier) is available now! 🔥🔥🔥
- **[2025.05.16]** Our MathCoder-VL is accepted to ACL 2025 Findings. 🔥🔥🔥
- **[2025.05.13]** 🔥🔥🔥 **[Seed1.5-VL](https://github.com/ByteDance-Seed/Seed1.5-VL)** achieves **68.7%** on MATH-Vision! 🎉 Congratulations!
- **[2025.04.11]** 💥 **Kimi-VL-A3B-Thinking achieves strong multimodal reasoning with just 2.8B LLM activated parameters!** Congratulations! See the full [leaderboard](https://mathllm.github.io/mathvision/#leaderboard).
- **[2025.04.10]** 🔥 **SenseNova V6 Reasoner** achieves **55.39%** on MATH-Vision! 🎉 Congratulations!
- **[2025.04.05]** 💥 **Step R1-V-Mini 🥇 Sets New SOTA on MATH-V with 56.6%!** See the full [leaderboard](https://mathllm.github.io/mathvision/#leaderboard).
- **[2025.03.10]** 💥 **Kimi k1.6 Preview Sets New SOTA on MATH-V with 53.29%!** See the full [leaderboard](https://mathllm.github.io/mathvision/#leaderboard).
- **[2025.02.28]** 💥 **Doubao-1.5-pro Sets New SOTA on MATH-V with 48.62%!** Read more on the [Doubao blog](https://team.doubao.com/zh/special/doubao_1_5_pro).
- **[2025.01.26]** 🚀 [Qwen2.5-VL-72B](http://qwenlm.github.io/blog/qwen2.5-vl/) achieves **38.1%**, establishing itself as the best-performing one in open-sourced models. 🎉 Congratulations!
- **[2025.01.22]** 💥 **Kimi k1.5  achieves new SOTA** on MATH-Vision with **38.6%**! Learn more at the [Kimi k1.5 report](https://arxiv.org/pdf/2501.12599).
- **[2024-09-27]** **MATH-V** is accepted by NeurIPS DB Track, 2024! 🎉🎉🎉
- **[2024-08-29]** 🔥 Qwen2-VL-72B achieves new open-sourced SOTA on MATH-Vision with 25.9! 🎉 Congratulations! Learn more at the [Qwen2-VL blog](https://qwenlm.github.io/blog/qwen2-vl/).
- **[2024-07-19]** [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) now supports **MATH-V**, utilizing LLMs for more accurate answer extraction!🔥
- **[2024-05-19]** OpenAI's **GPT-4o** scores **30.39%** on **MATH-V**, considerable advancement in short time! 💥
- **[2024-03-01]** **InternVL-Chat-V1-2-Plus** achieves **16.97%**, establishing itself as the new best-performing open-sourced model. 🎉 Congratulations!
- **[2024-02-23]** Our dataset is now accessible at [huggingface](https://huggingface.co/datasets/MathLLMs/MathVision).
- **[2024-02-22]** The top-performing model, **GPT-4V** only scores **23.98%** on **MATH-V**, while human performance is around **70%**.
- **[2024-02-22]** Our paper is now accessible at [ArXiv Paper](https://arxiv.org/abs/2402.14804).

## 👀 Introduction

Recent advancements in Large Multimodal Models (LMMs) have shown promising results in mathematical reasoning within visual contexts, with models approaching human-level performance on existing benchmarks such as MathVista. However, we observe significant limitations in the diversity of questions and breadth of subjects covered by these benchmarks. To address this issue, we present the MATH-Vision (MATH-V) dataset, a meticulously curated collection of 3,040 high-quality mathematical problems with visual contexts sourced from real math competitions. Spanning 16 distinct mathematical disciplines and graded across 5 levels of difficulty, our dataset provides a comprehensive and diverse set of challenges for evaluating the mathematical reasoning abilities of LMMs.


<p align="center">
    <img src="https://raw.githubusercontent.com/mathvision-cuhk/MathVision/main/assets/figures/figure1_new.png" width="66%">  The accuracies of four prominent Large Multimodal Models (LMMs), random chance, and human <br>
performance are evaluated on our proposed <b>MATH-Vision (MATH-V)</b> across 16 subjects.
</p>
<br>
Through extensive experimentation, we unveil a notable performance gap between current LMMs and human performance on MATH-V, underscoring the imperative for further advancements in LMMs.

You can refer to the [project homepage](https://mathvision-cuhk.github.io/) for more details.

## 🏆 Leaderboard

The leaderboard is available [here](https://mathvision-cuhk.github.io/#leaderboard).

We are commmitted to maintain this dataset and leaderboard in the long run to ensure its quality!
🔔 If you find any mistakes, please paste the question_id to the issue page, we will modify it accordingly.

## 📐 Dataset Examples

Some examples of MATH-V on three subjects: analytic geometry, topology, and graph theory.

<details>
<summary>Analytic geometry</summary><p align="center">
    <img src="https://raw.githubusercontent.com/mathvision-cuhk/MathVision/main/assets/examples/exam_analytic_geo.png" width="60%"> <br>
</p></details>

<details>
<summary>Topology</summary><p align="center">
    <img src="https://raw.githubusercontent.com/mathvision-cuhk/MathVision/main/assets/examples/exam_topology.png" width="60%"> <br>
</p></details>

<details>
<summary>Graph Geometry</summary><p align="center">
    <img src="https://raw.githubusercontent.com/mathvision-cuhk/MathVision/main/assets/examples/exam_graph.png" width="60%"> <br>
</p></details>



## 📑 Citation

If you find this benchmark useful in your research, please consider citing this BibTex:

```
@inproceedings{
wang2024measuring,
title={Measuring Multimodal Mathematical Reasoning with MATH-Vision Dataset},
author={Ke Wang and Junting Pan and Weikang Shi and Zimu Lu and Houxing Ren and Aojun Zhou and Mingjie Zhan and Hongsheng Li},
booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2024},
url={https://openreview.net/forum?id=QWTCcxMpPA}
}

@inproceedings{
wang2025mathcodervl,
title={MathCoder-{VL}: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning},
author={Ke Wang and Junting Pan and Linda Wei and Aojun Zhou and Weikang Shi and Zimu Lu and Han Xiao and Yunqiao Yang and Houxing Ren and Mingjie Zhan and Hongsheng Li},
booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics},
year={2025},
url={https://openreview.net/forum?id=nuvtX1imAb}
}
```
