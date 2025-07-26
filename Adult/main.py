import argparse
import json
import time
import numpy as np
import pandas as pd
from utils import load_adult_dataset, add_laplace_noise, perturb_weights
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from diffprivlib.models import LogisticRegression as DPLogReg
from diffprivlib.models import DecisionTreeClassifier as DPDecisionTree
import os

def run_experiment(config):
    print("[Adult Experiment]")
    print("Loading Adult dataset...")
    x_train, x_test, y_train, y_test, x_train_mm, x_test_mm, bounds = load_adult_dataset()

    epsilons = config['epsilons']
    methods = config['methods']
    models = ["logreg", "dt"]

    results = []

    # Baseline Models
    if "logreg" in models:
        print("\nRunning Baseline Logistic Regression...")
        start = time.time()
        baseline_lr = LogisticRegression(max_iter=1000).fit(x_train, y_train)
        y_pred_lr = baseline_lr.predict(x_test)
        acc = accuracy_score(y_test, y_pred_lr)
        f1 = f1_score(y_test, y_pred_lr)
        elapsed = time.time() - start
        results.append(["Baseline-LogReg", "N/A", acc, f1, elapsed])
        print(f"Baseline LR: accuracy={acc:.4f}, f1-score={f1:.4f}, runtime={elapsed:.2f}s")

    if "dt" in models:
        print("\nRunning Baseline Decision Tree...")
        start = time.time()
        baseline_dt = DecisionTreeClassifier().fit(x_train, y_train)
        y_pred_dt = baseline_dt.predict(x_test)
        acc = accuracy_score(y_test, y_pred_dt)
        f1 = f1_score(y_test, y_pred_dt)
        elapsed = time.time() - start
        results.append(["Baseline-DT", "N/A", acc, f1, elapsed])
        print(f"Baseline DT: accuracy={acc:.4f}, f1-score={f1:.4f}, runtime={elapsed:.2f}s")

    # Input Perturbation
    if "input" in methods:
        print("\n[Input Perturbation]")
        for eps in epsilons:
            # Logistic Regression
            start = time.time()
            noisy = add_laplace_noise(x_train, eps)
            lr = LogisticRegression(max_iter=1000).fit(noisy, y_train)
            y_pred = lr.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            elapsed = time.time() - start
            results.append([f"Input-LogReg", eps, acc, f1, elapsed])
            print(f"[Input-LR] epsilon={eps} accuracy={acc:.4f} f1-score={f1:.4f} runtime={elapsed:.2f}s")

            # Decision Tree
            start = time.time()
            dt = DecisionTreeClassifier().fit(noisy, y_train)
            y_pred = dt.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            elapsed = time.time() - start
            results.append([f"Input-DT", eps, acc, f1, elapsed])
            print(f"[Input-DT] epsilon={eps} accuracy={acc:.4f} f1-score={f1:.4f} runtime={elapsed:.2f}s")

    # Internal Perturbation
    if "internal" in methods:
        print("\n[Internal Perturbation]")
        for eps in epsilons:
            # Logistic Regression
            start = time.time()
            dp_lr = DPLogReg(epsilon=eps, data_norm=10.0).fit(x_train, y_train)
            y_pred = dp_lr.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            elapsed = time.time() - start
            results.append([f"Internal-LogReg", eps, acc, f1, elapsed])
            print(f"[Internal-LR] epsilon={eps} accuracy={acc:.4f} f1-score={f1:.4f} runtime={elapsed:.2f}s")

            # Decision Tree
            start = time.time()
            dp_dt = DPDecisionTree(epsilon=eps, bounds=bounds).fit(x_train_mm, y_train)
            y_pred = dp_dt.predict(x_test_mm)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            elapsed = time.time() - start
            results.append([f"Internal-DT", eps, acc, f1, elapsed])
            print(f"[Internal-DT] epsilon={eps} accuracy={acc:.4f} f1-score={f1:.4f} runtime={elapsed:.2f}s")

    # Output Perturbation
    if "output" in methods:
        print("\n[Output Perturbation]")
        for eps in epsilons:
            start = time.time()
            clf = LogisticRegression(max_iter=1000).fit(x_train, y_train)
            clf = perturb_weights(clf, epsilon=eps)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            elapsed = time.time() - start
            results.append([f"Output-LogReg", eps, acc, f1, elapsed])
            print(f"[Output-LR] epsilon={eps} accuracy={acc:.4f} f1-score={f1:.4f} runtime={elapsed:.2f}s")

    # Save Results
    df = pd.DataFrame(results, columns=["Method", "Epsilon", "Accuracy", "F1", "Runtime"])
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/results_adult.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential Privacy Experiment: Adult Dataset")
    parser.add_argument("--config", type=str, default="config_adult.json", help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    run_experiment(config)
