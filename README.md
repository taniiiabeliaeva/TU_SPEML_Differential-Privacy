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

## Getting Started

1. **Install dependencies**
   Run the following to install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Notebooks**
   Open the notebooks in `approaches_experiments/` to reproduce the results:

   * `adult.ipynb`: Differential privacy experiments on structured tabular data
   * `preprocess_mnist.ipynb`: Differential privacy applied to image classification
