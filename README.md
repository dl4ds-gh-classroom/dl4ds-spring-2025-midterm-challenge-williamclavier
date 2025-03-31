[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xnB1OI0j)
# DS542 Deep Learning for Data Science -- Spring 2025 Midterm Challenge

## Overview

This repository contains the code for the midterm challenge of the course DS542 Deep Learning for Data Science.

The challenge is in three parts:
1. **Part 1 -- Simple CNN:** Define a relatively simple CNN model and train it on the CIFAR-100 dataset to
    get a complete pipeline and establish baseline performance.
2. **Part 2 -- More Sophisticated CNN Models:** Use a more sophisticated model, including predefined models from torchvision
   to train and evaluate on CIFAR-100.
3. **Part 3 -- Transfer Learning from a Pretrained Model:** Pretrain a model, or use one of the pretrained models from torchvision, and
   fine-tune it on CIFAR-100. Try to beat the best benchmark performance on the leaderboard.

All your models should be built from linear and convoultional layers, as well as pooling, etc. We haven't covered Transformers yet,
so don't use Transformer architectures.

There is example starter template in `starter_code.py` which includes evaluation and submissions generation code. We suggest
you copy and revise that code for each of the three parts above. In other words, your repo should have (at least) the three
files, one for each part described above, as well as any supporting files.

For each part, submit the results to the Kaggle [leaderboard](https://www.kaggle.com/t/3551aa4f562f4b79b93204b11ae640b4).

Your best result needs beat the best benchmark performance of 0.397 on the leaderboard.

Use Weights and Biases experiment tracking tool to track your experiments. Create
a free student account at [WandB](https://wandb.ai). The starter code is already
instrumented for WandB, so it will start tracking experiments right away.

You can write your report using the WandB Reports UI if you wish.

## Data

You will start with the CIFAR-100 dataset, which is downloaded and installed the
first time your successfully run the sample code, `starter_code.py`.

It should install into the `data/cifar-100-python` directory.

We also have the challenge images in `data/ood-test` directory. Those are used
to make predictions on the challenge images with your model and produce the 
submission file.

## Setup

Fork this repository to your GitHub account and clone it to your local machine
or to the SCC.

On MacOS and Linux, you can create a virtual environment and install the
dependencies with the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Report

In addition to the code, we require a short report that includes:

* **AI Disclosure:** You are allowed to use AI assistance for this assignment, but you are required to:
    * Explain how you used AI, e.g. Copilot, Cursor, ChatGPT, etc.
    * Enumerate in detail which parts of the code were written by you and which were written with AI assistance.
    * Have detailed code comments explaining what every part of your code does. This can be in the codebase itself.
    * **Failure to disclose how you used AI may result in a score of 0 for the assignment.**
* **Model Description:** Detailed explanation of the chosen architecture, including justifications for design choices.
* **Hyperparameter Tuning:** Description of the hyperparameter search process and the final chosen values.
* **Regularization Techniques:** Explanation of the regularization methods used and their impact.
* **Data Augmentation Strategy:** Description of the data augmentation techniques used.
* **Results Analysis:** Discussion of the results, including strengths and weaknesses of the model, and potential areas for improvement.
* **Experiment Tracking Summary:**  Include screenshots or summaries from the experiment tracking tool.
  You can use the WandB Reports UI to create a report as well.

## Grading Rubric

The grading rubric is as follows:

* **Code Quality (30%):**
    * Correctness of implementation.
    * Readability and organization of code.
    * Use of PyTorch best practices.
    * Efficiency of data loading and processing.
* **Model Performance (40%):**
    * Performance on the primary evaluation metric.
    * Ranking on the leaderboard of at least above 0.397
    * List the leaderboard performance, identifier and username for the best scores for each of the three parts of the assignment.
* **Experiment Tracking and Report (30%):**
    * Comprehensive AI disclosure statement.
    * Completeness and clarity of the report.
    * Thoroughness of experiment tracking.
    * Justification of design choices.
    * Analysis of results.
    * Ablation study (if included).

## Bonus Points (Optional)

The top 10 students on the Private leaderboard will receive bonus points.
