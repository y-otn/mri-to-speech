# mri-to-speech

## Pre-requisites

1. NVIDIA GPU + CUDA cuDNN
2. TensorFlow NGC Container ~= nvcr.io/nvidia/tensorflow:23.02-tf2-py3

## Setup

1. Clone this repository: `git clone https://github.com/y-otn/mri-to-speech.git`
2. Install missing packages:

    ```console
    apt update
    apt install libsndfile1
    ```

3. Install python requirements: `pip install -r requirements.txt`

## Training & Inference

1. `python main.py`
