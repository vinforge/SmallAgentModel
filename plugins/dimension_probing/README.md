# Code for paper “Human-like object concept representations emerge naturally in multimodal large language models". Nature Machine Intelligence, 2025
> - authors: Changde Du, Kaicheng Fu, Bincheng Wen, Yi Sun, Jie Peng, Wei Wei, Ying Gao, Shengpei Wang, Chuncheng Zhang, Jinpeng Li, Shuang Qiu, Le Chang, Huiguang He.

## Overview

This repository consists of three parts:
1. `data_collecting`: code for collecting triplet odd-one-out data using different models (LLMs, MLLMs) or features extracted from DNNs or brain ROIs.
2. `model_training`: code for learning interpretable embeddings from the collected triplet odd-one-out recordings (using [SPoSE](https://github.com/ViCCo-Group/SPoSE) method). 
3. `analysis_and_figure_drawing`: code and experiments for reproducing the main results of the paper. 

## Environment Setup

### Step 1: Clone the Repository

Clone the repository and navigate to the project root:

```bash
git clone https://github.com/ChangdeDu/LLMs_core_dimensions.git
cd LLMs_core_dimensions
```

### Step 2: Install Dependencies

1. The code uses Python 3.8, MATLAB R2021b, PyTorch 1.6.0.
2. Install PyTorch: `pip install pytorch` or `conda install pytorch torchvision -c pytorch` (the latter is recommended if you use Anaconda)
3. Install Python dependencies: `pip install -r requirements.txt`

## Odd-One-Out Data Collecting

```bash
cd data_collecting
```

The code used for odd-one-out data collecting is given. But you can skip this step and download the data we collected directly from [OSF](https://osf.io/qn5uv/).

## Learning Interpretable Embeddings

We learn interpretable embeddings from the collected triplet odd-one-out recordings. The data can come from:

1. Humans: behavioral experiments with humans.
2. LLMs/MLLMs: behavioral experiments with LLMs/MLLMs.
3. DNNs: simulating triplet choices from DNN activations (based on cosine distance)
4. Brain ROIs: simulating triplet choices from brain ROI activations (e.g., FFA, EBA, etc.).

```bash
cd model_training
```

The code is copied from [SPoSE's official repository](https://github.com/ViCCo-Group/SPoSE). Please see the [README](../model_training/SPoSE/README.md) file provided by the original author for code details. 


You should store the training and test data in the `model_training/SPoSE/data` folder before running it.

The path to the `model_training/SPoSE` folder should be as follows:

```
SPoSE
├── data
│   └── chatgpt3.5_things
│       └── train_90.txt
│       └── test_10.txt
│   └── gemini_pro_vision_things
│   └── human_things
│   └── simclr_things
│   └── vgg16_things
│   └── ...
├── models
├── results
├── get_spose_embedding.py
├── ...
```

We have written the model training instructions and configuration in a separate file, and you can run the SPoSE algorithm directly by executing the following commands to learn interpretable low-dimensional embeddings.

```bash
python get_spose_embedding.py
```

The learned low-dimensional embedding is stored in the results folder, named `weights_sorted.npy`, and the path information shows the specific parameter configuration of the learning process. For example, SimCLR' low-dimensional embedding will be stored in `model_training/SPoSE/results/simclr/100d/0.007/seed242/weights_sorted.npy`. Here `100d` represents the initialization dimension, `0.007` represents the learning rate, and `seed242` represents the random seed.

## Main Experiments of the Paper and Figure Drawing


Before running the experiment and drawing the figures, you should download the required data from [OSF](https://osf.io/qn5uv/) and extract it to the `analysis_and_figure_drawing/data` folder:

```bash
cd analysis_and_figure_drawing
wget https://osf.io/download/qn5uv/ -O data.tar.gz
-xvf data.tar.gz
```
The path to the `analysis_and_figure_drawing` folder should be as follows:

```
analysis_and_figure_drawing
├── data
│   └── DNNs
│   └── Humans
│   └── LLMs
│   └── MLLMs
│   └── ROIs
│   └── variables
│   └── ...
├── GPT3.5vsGPT4
├── helper_functions
├── src
├── draw_figure2a_step1_get_RDM48_for_models.m
├── draw_figure2a_step2_r48_values_against_dims_retained_llm_mllm_human.m
├── ...
```

After that, you can run the relevant experimental analysis and figure drawing code one by one. Each important figure in the paper is associated with one (or more) code files, and the filename indicates the index of the corresponding figure. For example, if you want to reproduce Figure 2a, then you can run `draw_figure2a_step1_get_RDM48_for_models.m` and then `draw_figure2a_step2_r48_values_against_dims_retained_llm_mllm_human.m` in Matlab environment.

## Acknowledgments

This repository heavily uses the previous open source code, such as [SPoSE](https://github.com/ViCCo-Group/SPoSE), [Revealing the multidimensional mental representations of natural objects underlying human similarity judgments](https://osf.io/z2784/) and so on. We express our heartfelt thanks to the original authors.
