---
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: image
    dtype: image
  splits:
  - name: test
    num_bytes: 678377348
    num_examples: 765
  download_size: 678335644
  dataset_size: 678377348
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
task_categories:
- visual-question-answering
language:
- en
pretty_name: RealWorldQA
---

# RealWorldQA dataset

This is the benchmark dataset released by xAI along with the Grok-1.5 Vision [announcement](https://x.ai/blog/grok-1.5v). 
This benchmark is designed to evaluate basic real-world spatial understanding capabilities of multimodal models. 
While many of the examples in the current benchmark are relatively easy for humans, they often pose a challenge for frontier models.

This release of the RealWorldQA consists of 765 images, with a question and easily verifiable answer for each image. 
The dataset consists of anonymized images taken from vehicles, in addition to other real-world images.

## License

CC BY-ND 4.0