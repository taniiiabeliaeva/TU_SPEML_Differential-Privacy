# TU_SPEML_Differential-Privacy

This project explores and compares different approaches for achieving **Differential Privacy (DP)** in machine learning, with a focus on the trade-off between **privacy guarantees** and **model performance**.

## Goal

To experimentally evaluate three core differential privacy techniques:

* **Input Perturbation** – noise is added directly to input features
* **Internal Perturbation** – DP is enforced via algorithm internals (e.g., private optimization)
* **Output Perturbation** – noise is applied post-training to model outputs or parameters

The experiments measure the impact of each method on:

* **Model accuracy and F1-score**
* **Runtime efficiency**
* **Privacy-utility trade-off**

## Project Structure
```
TU_SPEML_Differential-Privacy/
│
├── MNIST/                        # MNIST dataset experiments
│   ├── main.py                   # MNIST DP experiment script
│   ├── config.json               # Config file for MNIST
│   ├── utils.py                  # Utility functions (load dataset, noise functions)
│   └── mnist.npz                 # MNIST dataset (included for reproducibility)
│
├── Adult/                        # Adult dataset experiments
│   ├── main.py                   # Adult DP experiment script
│   ├── config.json               # Config file for Adult dataset
│   └── utils.py                  # Preprocessing and helper functions
│
├── results/                      # Results
│   ├── results_mnist.csv
│   └── results_adult.csv
│
├── approaches_experiments/       # Jupyter notebooks for step-by-step analysis
│   ├── mnist.ipynb               # MNIST dataset experiments (input/internal/output perturbation)
│   ├── adult.ipynb               # Adult dataset experiments (input/internal/output perturbation)
│   └── mnist.npz                 # Local MNIST copy for notebooks
│
├── run.sh                        # Shell script to execute experiments
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

## Getting Started

### 1. Run  
You can execute all experiments (MNIST + Adult) with a single command:
```
./run.sh
```
- Install required dependencies  
- Run MNIST experiments  
- Run Adult dataset experiments  
- Save results in results/

### 2. MacOS Users  
If you are using MacOS, open requirements.txt and uncomment the lines starting with # for mac-specific packages.
```
tensorflow-macos
tensorflow-metal
```
