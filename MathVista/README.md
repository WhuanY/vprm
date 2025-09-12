---
annotations_creators:
- expert-generated
- found
language_creators:
- expert-generated
- found
language:
- en
- zh
- fa
license: cc-by-sa-4.0
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- multiple-choice
- question-answering
- visual-question-answering
- text-classification
task_ids:
- multiple-choice-qa
- closed-domain-qa
- open-domain-qa
- visual-question-answering
- multi-class-classification
paperswithcode_id: mathvista
pretty_name: MathVista
tags:
- multi-modal-qa
- math-qa
- figure-qa
- geometry-qa
- math-word-problem
- textbook-qa
- vqa
- arithmetic-reasoning
- statistical-reasoning
- algebraic-reasoning
- geometry-reasoning
- numeric-common-sense
- scientific-reasoning
- logical-reasoning
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
- medical-image
- mathematics
- science
- chemistry
- biology
- physics
- engineering
- natural-science
configs:
- config_name: default
  data_files:
  - split: testmini
    path: data/testmini-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: pid
    dtype: string
  - name: question
    dtype: string
  - name: image
    dtype: string
  - name: decoded_image
    dtype: image
  - name: choices
    sequence: string
  - name: unit
    dtype: string
  - name: precision
    dtype: float64
  - name: answer
    dtype: string
  - name: question_type
    dtype: string
  - name: answer_type
    dtype: string
  - name: metadata
    struct:
    - name: category
      dtype: string
    - name: context
      dtype: string
    - name: grade
      dtype: string
    - name: img_height
      dtype: int64
    - name: img_width
      dtype: int64
    - name: language
      dtype: string
    - name: skills
      sequence: string
    - name: source
      dtype: string
    - name: split
      dtype: string
    - name: task
      dtype: string
  - name: query
    dtype: string
  splits:
  - name: testmini
    num_bytes: 142635198.0
    num_examples: 1000
  - name: test
    num_bytes: 648291350.22
    num_examples: 5141
  download_size: 885819490
  dataset_size: 790926548.22
---
# Dataset Card for MathVista

- [Dataset Description](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#dataset-description)
- [Paper Information](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#paper-information)
- [Dataset Examples](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#dataset-examples)
- [Leaderboard](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#leaderboard)
- [Dataset Usage](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#dataset-usage)
  - [Data Downloading](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#data-downloading)
  - [Data Format](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#data-format)
  - [Data Visualization](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#data-visualization)
  - [Data Source](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#data-source)
  - [Automatic Evaluation](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#automatic-evaluation)
- [License](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#license)
- [Citation](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/README.md#citation)

## Dataset Description

**MathVista** is a consolidated Mathematical reasoning benchmark within Visual contexts. It consists of **three newly created datasets, IQTest, FunctionQA, and PaperQA**, which address the missing visual domains and are tailored to evaluate logical reasoning on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures, respectively. It also incorporates **9 MathQA datasets** and **19 VQA datasets** from the literature, which significantly enrich the diversity and complexity of visual perception and mathematical reasoning challenges within our benchmark. In total, **MathVista** includes **6,141 examples** collected from **31 different datasets**.

## Paper Information

- Paper: https://arxiv.org/abs/2310.02255
- Code: https://github.com/lupantech/MathVista
- Project: https://mathvista.github.io/
- Visualization: https://mathvista.github.io/#visualization
- Leaderboard: https://mathvista.github.io/#leaderboard

## Dataset Examples

Examples of our newly annotated datasets: IQTest, FunctionQA, and PaperQA:

<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/our_new_3_datasets.png" style="zoom:40%;" />

<details>
<summary>🔍 Click to expand/collapse more examples</summary>

Examples of seven mathematical reasoning skills:

1. Arithmetic Reasoning

<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/skills/ari.png" style="zoom:40%;" />

2. Statistical Reasoning

<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/skills/sta.png" style="zoom:40%;" />

3. Algebraic Reasoning

<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/skills/alg.png" style="zoom:40%;" />

4. Geometry Reasoning

<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/skills/geo.png" style="zoom:40%;" />

5. Numeric common sense

<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/skills/num.png" style="zoom:40%;" />

6. Scientific Reasoning

<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/skills/sci.png" style="zoom:40%;" />

7. Logical Reasoning

<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/skills/log.png" style="zoom:40%;" />

</details>

## Leaderboard

🏆 The leaderboard for the *testmini* set (1,000 examples) is available [here](https://mathvista.github.io/#leaderboard).

🏆 The leaderboard for the *test* set (5,141 examples) and the automatic evaluation on [CodaLab](https://codalab.org/) are under construction. 

## Dataset Usage

### Data Downloading

All the data examples were divided into two subsets: *testmini* and *test*.

- **testmini**: 1,000 examples used for model development, validation, or for those with limited computing resources.
- **test**: 5,141 examples for standard evaluation. Notably, the answer labels for test will NOT be publicly released.

You can download this dataset by the following command (make sure that you have installed [Huggingface Datasets](https://huggingface.co/docs/datasets/quickstart)):

```python
from datasets import load_dataset

dataset = load_dataset("AI4Math/MathVista")
```

Here are some examples of how to access the downloaded dataset:

```python
# print the first example on the testmini set
print(dataset["testmini"][0])
print(dataset["testmini"][0]['pid']) # print the problem id 
print(dataset["testmini"][0]['question']) # print the question text 
print(dataset["testmini"][0]['query']) # print the query text
print(dataset["testmini"][0]['image']) # print the image path
print(dataset["testmini"][0]['answer']) # print the answer
dataset["testmini"][0]['decoded_image'] # display the image

# print the first example on the test set
print(dataset["test"][0])
```

### Data Format

The dataset is provided in json format and contains the following attributes:

```json
{
    "question": [string] The question text,
    "image": [string] A file path pointing to the associated image,
    "choices": [list] Choice options for multiple-choice problems. For free-form problems, this could be a 'none' value,
    "unit": [string] The unit associated with the answer, e.g., "m^2", "years". If no unit is relevant, it can be a 'none' value,
    "precision": [integer] The number of decimal places the answer should be rounded to,
    "answer": [string] The correct answer for the problem,
    "question_type": [string] The type of question: "multi_choice" or "free_form",
    "answer_type": [string] The format of the answer: "text", "integer", "float", or "list",
    "pid": [string] Problem ID, e.g., "1",
    "metadata": {
        "split": [string] Data split: "testmini" or "test",
        "language": [string] Question language: "English", "Chinese", or "Persian",
        "img_width": [integer] The width of the associated image in pixels,
        "img_height": [integer] The height of the associated image in pixels,
        "source": [string] The source dataset from which the problem was taken,
        "category": [string] The category of the problem: "math-targeted-vqa" or "general-vqa",
        "task": [string] The task of the problem, e.g., "geometry problem solving",
        "context": [string] The visual context type of the associated image,
        "grade": [string] The grade level of the problem, e.g., "high school",
        "skills": [list] A list of mathematical reasoning skills that the problem tests
		},
    "query": [string] the query text used as input (prompt) for the evaluation model
}
```

### Data Visualization

🎰 You can explore the dataset in an interactive way [here](https://mathvista.github.io/#visualization).

<details>
<summary>Click to expand/collapse the visualization page screeshot.</summary>
<img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/data_visualizer.png" style="zoom:40%;" />
</details>

### Data Source

The **MathVista** dataset is derived from three newly collected datasets: IQTest, FunctionQA, and Paper, as well as 28 other source datasets. Details can be found in the [source.json](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/source.json) file. All these source datasets have been preprocessed and labeled for evaluation purposes.

### Automatic Evaluation

🔔 To automatically evaluate a model on the dataset, please refer to our GitHub repository [here](https://github.com/lupantech/MathVista/tree/main).

## License

The new contributions to our dataset are distributed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license, including

- The creation of three datasets: IQTest, FunctionQA, and Paper;
- The filtering and cleaning of source datasets;
- The standard formalization of instances for evaluation purposes;
- The annotations of metadata.

The copyright of the images and the questions belongs to the original authors, and the source of every image and original question can be found in the `metadata` field and in the [source.json](https://huggingface.co/datasets/AI4Math/MathVista/blob/main/source.json) file. Alongside this license, the following conditions apply:

- **Purpose:** The dataset was primarily designed for use as a test set.
- **Commercial Use:** The dataset can be used commercially as a test set, but using it as a training set is prohibited. By accessing or using this dataset, you acknowledge and agree to abide by these terms in conjunction with the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.

## Citation

If you use the **MathVista** dataset in your work, please kindly cite the paper using this BibTeX:

```
@inproceedings{lu2024mathvista,
  author = {Lu, Pan and Bansal, Hritik and Xia, Tony and Liu, Jiacheng and Li, Chunyuan and Hajishirzi, Hannaneh and Cheng, Hao and Chang, Kai-Wei and Galley, Michel and Gao, Jianfeng},
  title = {MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2024}
}
```