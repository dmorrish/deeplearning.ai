import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from joblib import dump
from custom_nn_implementation import Layer, Sequential


def normalize(X):
    (m, n) = X.shape
    means = np.mean(X, axis=0)
    stds = np.clip(np.std(X, axis=0), 1, 1e100)

    return (X - means) / stds


np.set_printoptions(precision=12)

X = np.load("./X.npy")
Y = np.load("./Y.npy")

X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size=0.2)

epochs = 1500
alpha = 3e-1
alpha_tf = 1e-6
adam = 0.003
lambda_ = 0.00
epsilon = 1e-6
verbose = False

Xn_train = normalize(X_train)
Xn_cv = normalize(X_cv)

model = Sequential(
    [
        Layer(units=25, activation="relu", name="Input"),
        # Layer(units=10, activation="relu"),
        Layer(units=10, activation="linear", name="Output")
    ],
    cost_function="sparse_categorical_crossentropy",
    from_logits=True
)

print(f"\nTraining neural network with the following configuration:")
print(f"=========================================================")
for layer in model.layers:
    print(f"  {layer.name}: ")
    print(f"    units: {layer.units}")
    print(f"    activation: {layer.activation.__name__}")
print(f"  alpha: {alpha}")
print(f"  adaptive_moment: {adam}")
print(f"  cost_function: {model.cost_function.__name__}")
print(f"=========================================================")
start = datetime.now()
model.train(Xn_train, Y_train, alpha, iterations=epochs, epsilon=epsilon, adam=adam, lambda_=lambda_, verbose=verbose)
end = datetime.now()
print(f"=========================================================")
output = np.argmax(model.propagate(Xn_cv), axis=1)
truth = Y_cv.T[0].astype(int)
success = np.sum((output == truth).astype(int))
print(f"Output of test points:\n{output}")
print(f"Expected output of test points:\n{truth}\n\n")
print(f"Differences:\n{(output != truth)}")
print(f"Success rate: {success} / {np.size(output)} = {success / np.size(output)}")
print(f"Training time: {end - start}")
dump(model, "nn_digits_custom.joblib")
