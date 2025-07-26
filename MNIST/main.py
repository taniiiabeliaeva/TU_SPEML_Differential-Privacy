import argparse
import json
import time
import numpy as np
import pandas as pd
from utils import load_mnist, add_laplace_noise, perturb_weights, build_cnn_model
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from diffprivlib.models import LogisticRegression as DPLogisticRegression
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def run_experiment(config):
    # Load MNIST dataset
    print("[MNIST Experiment]")
    print("Loading MNIST dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    x_train_flat = x_train.reshape(-1, 28 * 28)
    x_test_flat = x_test.reshape(-1, 28 * 28)

    epsilons = config['epsilons']
    methods = config['methods']
    models = config.get('models', ['logreg'])  # ["logreg", "cnn"]

    # One-hot encoding for CNN
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    results = []

    # Running Baseline
    if "logreg" in models:
        print("\nRunning baseline Logistic Regression...")
        start = time.time()
        clf_base = LogisticRegression(max_iter=1000)
        clf_base.fit(x_train_flat, y_train)
        y_pred = clf_base.predict(x_test_flat)
        baseline_acc = accuracy_score(y_test, y_pred)
        baseline_f1 = f1_score(y_test, y_pred, average='weighted')
        elapsed = time.time() - start
        results.append(["Baseline-LogReg", "N/A", baseline_acc, baseline_f1, elapsed])
        print(f"LogReg Baseline: accuracy={baseline_acc:.4f}, f1-score={baseline_f1:.4f}, runtime={elapsed:.2f}s")

    if "cnn" in models:
        print("\nRunning baseline CNN...")
        start = time.time()
        cnn = build_cnn_model()
        cnn.fit(x_train[..., None], y_train_cat, epochs=3, batch_size=64, verbose=0)
        loss, test_acc = cnn.evaluate(x_test[..., None], y_test_cat, verbose=0)
        elapsed = time.time() - start
        y_pred_cnn = cnn.predict(x_test[..., None], verbose=0)
        y_pred_labels = np.argmax(y_pred_cnn, axis=1)
        cnn_f1 = f1_score(y_test, y_pred_labels, average='weighted')
        results.append(["Baseline-CNN", "N/A", test_acc, cnn_f1, elapsed])
        print(f"CNN Baseline: accuracy={test_acc:.4f}, f1-score={cnn_f1:.4f}, runtime={elapsed:.2f}s")


    # Methods
    if "input" in methods:
        if "logreg" in models:
            print("\n[Input Perturbation (Logistic Regression)]")
            print("Epsilon | Accuracy | F1-Score | Runtime")
            for eps in epsilons:
                start = time.time()
                x_train_perturbed = add_laplace_noise(x_train_flat, eps)
                clf = LogisticRegression(max_iter=1000)
                clf.fit(x_train_perturbed, y_train)
                y_pred = clf.predict(x_test_flat)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                elapsed = time.time() - start
                results.append(["Input-LogReg", eps, acc, f1, elapsed])
                print(eps, acc, f1, elapsed)

        if "cnn" in models:
            print("\n[Input Perturbation (CNN)]")
            print("Epsilon | Accuracy | F1-Score | Runtime")
            for eps in epsilons:
                start = time.time()
                x_train_perturbed = add_laplace_noise(x_train, eps)
                cnn = build_cnn_model()
                cnn.fit(x_train_perturbed[..., None], y_train_cat, epochs=3, batch_size=64, verbose=0)
                loss, acc = cnn.evaluate(x_test[..., None], y_test_cat, verbose=0)
                elapsed = time.time() - start
                y_pred_cnn = cnn.predict(x_test[..., None], verbose=0)
                y_pred_labels = np.argmax(y_pred_cnn, axis=1)
                f1 = f1_score(y_test, y_pred_labels, average='weighted')
                results.append(["Input-CNN", eps, acc, f1, elapsed])
                print(eps, acc, f1, elapsed)
            
    if "internal" in methods:
        print("\n[Internal Perturbation]")
        print("Epsilon | Accuracy | F1-Score | Runtime")
        for eps in epsilons:
            start = time.time()
            clf = DPLogisticRegression(epsilon=eps, data_norm=28.0)
            clf.fit(x_train_flat, y_train)
            y_pred = clf.predict(x_test_flat)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            elapsed = time.time() - start
            results.append(["Internal", eps, acc, f1, elapsed])
            print(eps, acc, f1, elapsed)

    if "output" in methods:
        print("\n[Output Perturbation]")
        print("Epsilon | Accuracy | F1-Score | Runtime")
        for eps in epsilons:
            start = time.time()
            clf = LogisticRegression(max_iter=1000)
            clf.fit(x_train_flat, y_train)
            clf = perturb_weights(clf, eps)
            y_pred = clf.predict(x_test_flat)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            elapsed = time.time() - start
            results.append(["Output", eps, acc, f1, elapsed])
            print(eps, acc, f1, elapsed)

    # Save results
    df = pd.DataFrame(results, columns=["Method", "Epsilon", "Accuracy", "F1", "Runtime"])
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/results_mnist.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Differential Privacy Experiment: MNIST")
    parser.add_argument("--config", type=str, default="MNIST/config.json", help="Path to config file")
    parser.add_argument("--epsilon", type=float, help="Epsilon value")
    parser.add_argument("--method", type=str, help="Method (input/internal/output)")
    parser.add_argument("--model", type=str, help="Model (logreg/cnn)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Override if provided
    if args.epsilon:
        config['epsilons'] = [args.epsilon]
    if args.method:
        config['methods'] = [args.method]
    if args.model:
        config['models'] = [args.model]

    run_experiment(config)

