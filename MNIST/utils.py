import urllib.request
import numpy as np
from tensorflow.keras import layers, models

# Loading MNIST dataset
def load_mnist():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    filename = "mnist.npz"
    urllib.request.urlretrieve(url, filename)

    with np.load(filename) as data:
        x_train = data["x_train"].astype("float32") / 255.
        y_train = data["y_train"]
        x_test = data["x_test"].astype("float32") / 255.
        y_test = data["y_test"]
    return x_train, y_train, x_test, y_test

# Laplace Noise (for Input Perturbation)
def add_laplace_noise(data, epsilon, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=data.shape)
    return np.clip(data + noise, 0.0, 1.0)

# Add noise to weights (for Output Perturbation)
def perturb_weights(model, epsilon, sensitivity=1.0):
    scale = sensitivity / epsilon
    model.coef_ += np.random.laplace(loc=0.0, scale=scale, size=model.coef_.shape)
    model.intercept_ += np.random.laplace(loc=0.0, scale=scale, size=model.intercept_.shape)
    return model

# Set CNN
def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
