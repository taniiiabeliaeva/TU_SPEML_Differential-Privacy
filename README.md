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
Exercise2_Group20/
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
chmod +x run.sh
./run.sh
```
- Install required dependencies  
- Run MNIST experiments  
- Run Adult dataset experiments  
- Save results in results/

### 2. Parameter Usage
Both MNIST/main.py and Adult/main.py support command-line overrides for configuration.
If you do not provide arguments, defaults from config.json will be used.

#### Available Parameters
|Argument	| Description |	Example|
|------------|----------|---------------------------------|
| --config	|Path to config file (optional)	| --config MNIST/config.json|
| --epsilon	|Override epsilon value (float)	| --epsilon 1.0 |
| --method	|input, internal, output (DP Method)	| --method input |
| --model	|MNIST: logreg/cnn (MNIST), Adult: logreg/dt	| --model cnn |

#### Examples
Run MNIST with custom epsilon and method:
```
python MNIST/main.py --epsilon 1.0 --method input
```
Run Adult with Decision Tree and internal perturbation:
```
python Adult/main.py --method internal --model dt --epsilon 5.0
```
Run using config.json but override epsilon:
```
python MNIST/main.py --config MNIST/config.json --epsilon 0.5
```


### 3. MacOS Users  
If you are using MacOS, open requirements.txt and uncomment the lines starting with # for mac-specific packages.
```
tensorflow-macos
tensorflow-metal
```
