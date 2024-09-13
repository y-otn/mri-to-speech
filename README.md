# mri-to-speech

This repository provides code for performing speech synthesis and F0 estimation from MRI videos.

Visit our [demo website](https://y-otn.github.io/mri-to-speech-demo/) to listen to audio samples. Multi-speaker samples are also available [here](https://y-otn.github.io/mri-to-speech-multi-demo/).

## Requirements

1. NVIDIA GPU with NVIDIA Driver and Docker
2. TensorFlow NGC Container ~= nvcr.io/nvidia/tensorflow:23.02-tf2-py3

## Setup

1. Clone this repository: `git clone https://github.com/y-otn/mri-to-speech.git`
2. Install the required packages:

    ```console
    apt update
    apt install libsndfile1
    ```

3. Install Python dependencies: `pip install -r requirements.txt`

## Training & Inference

1. `python run.py` (for speech synthesis)
2. `python run_f0.py` (for F0 estimation)

## Note

The code will be refactored in future updates to improve readability.
