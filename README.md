# Agentic2Accelerate
Deep Reinforcement Learning-Based Acceleration Model for Recovery Algorithm

This repository contains the implementation of a deep reinforcement learning framework designed to accelerate recovery algorithms in **Single-Pixel Imaging (SPI)**.

## 🚀 Getting Started

Follow these steps to set up the environment and run the tests

### 1. Installation
First, install the necessary dependencies using the provided requirements file:
```bash
pip install -r requirements.txt
```

### 2. Dependencies
This project use the **DeepInverse** library. We use its specialized operators and framework to integrate deep learning models with physical inverse problems in computational imaging.
- **Reference**: [DeepInverse GitHub](https://github.com/deepinv/deepinv)

### 3. Testing the Project
To evaluate the acceleration model and the algorithm selection policy for PGD, run the following script:
```bash
python test_algo_selector_pnp_SPC.py
```
