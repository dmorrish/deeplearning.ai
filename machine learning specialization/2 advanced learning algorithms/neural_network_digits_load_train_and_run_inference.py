from joblib import load
import numpy as np
from custom_nn_implementation import Layer, Sequential
from sklearn.model_selection import train_test_split


def normalize(X):
    (m, n) = X.shape
    means = np.mean(X, axis=0)
    stds = np.clip(np.std(X, axis=0), 1, 1e100)

    return (X - means) / stds


model = load("nn_digits_custom.joblib")

X = np.load("./X.npy")
Y = np.load("./Y.npy")

X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size=0.2)
Xn_train = normalize(X_train)
Xn_cv = normalize(X_cv)

epochs = 1000
alpha = 3e-1
alpha_tf = 1e-6
adam = 0.003
lambda_ = 0.00
epsilon = 1e-6
verbose = False

model.train(Xn_train, Y_train, alpha, iterations=epochs, epsilon=epsilon, adam=adam, lambda_=lambda_, verbose=verbose)

output = np.argmax(model.propagate(Xn_cv), axis=1)
truth = Y_cv.T[0].astype(int)
success = np.sum((output == truth).astype(int))
print(f"Output of test points:\n{output}")
print(f"Expected output of test points:\n{truth}\n\n")
print(f"Differences:\n{(output != truth)}")
print(f"Success rate: {success} / {np.size(output)} = {success / np.size(output)}")
