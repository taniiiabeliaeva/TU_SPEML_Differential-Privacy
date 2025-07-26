import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_adult_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    df = pd.read_csv(url, names=columns, sep=',', skipinitialspace=True, na_values="?")
    df = df.dropna()

    # Encode categoricals
    categorical = df.select_dtypes(include='object').columns
    df[categorical] = df[categorical].apply(LabelEncoder().fit_transform)

    X = df.drop("income", axis=1)
    y = df["income"]
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # For DP trees: scale to [0,1] and save bounds
    X_minmax = MinMaxScaler().fit_transform(X)
    x_train_mm, x_test_mm, _, _ = train_test_split(X_minmax, y, test_size=0.2, random_state=42)
    bounds = (np.zeros(X.shape[1]), np.ones(X.shape[1]))

    return x_train, x_test, y_train, y_test, x_train_mm, x_test_mm, bounds

def add_laplace_noise(data, epsilon, sensitivity=1.0):
    scale = sensitivity / epsilon
    return data + np.random.laplace(loc=0, scale=scale, size=data.shape)

def perturb_weights(model, epsilon, sensitivity=1.0):
    scale = sensitivity / epsilon
    model.coef_ += np.random.laplace(0, scale, model.coef_.shape)
    model.intercept_ += np.random.laplace(0, scale, model.intercept_.shape)
    return model
