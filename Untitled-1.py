# data/download_mnist.py

import os
import tensorflow as tf
import numpy as np

def save_mnist(data_dir="data/mnist"):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, "x_train.npy"), x_train)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "x_test.npy"), x_test)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    print(f"MNIST data saved to {data_dir}")

if __name__ == "__main__":
    save_mnist()
