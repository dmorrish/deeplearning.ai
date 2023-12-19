import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from datetime import datetime


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5          # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))


class Layer():
    W = None
    B = None
    units = None
    activation = None
    A_in = None
    A_out = None
    dJ_dW = None
    dJ_dW_prev = None
    dJ_dB = None
    dJ_dB_prev = None
    alpha = None
    alpha_W = None
    alpha_B = None
    dir_change_W_prev = None
    dir_change_B_prev = None
    name = None

    def __init__(self, name: str = None, units: int = 1, activation: str = "linear"):
        if units <= 0:
            print("ERROR: Units must be a positive integer.")
            return
        self.units = units

        if activation == "sigmoid":
            self.activation = self.sigmoid
        elif activation == "relu":
            self.activation = self.relu
        elif activation == "linear":
            self.activation = self.linear
        elif activation == "softmax":
            self.activation = self.softmax
        else:
            print("Unknown activation. Defaulting to linear.")
            self.activation = self.linear

        self.name = name

    def init_W_B(self, num_inputs: int):
        if self.W is not None or self.B is not None:
            print("ERROR: W and B were already initialized.")
            return
        if self.units is None:
            print("ERROR: Unit count was never initialized.")
            return
        if num_inputs < 1:
            print("ERROR: The layer must have at least one input.")
            return
        rng = np.random.default_rng(2)
        self.W = rng.random((num_inputs, self.units))
        self.B = rng.random((self.units,))

    def propagate(self, A_in):
        self.A_in = A_in
        Z = np.matmul(A_in, self.W) + self.B
        A_out = self.activation(Z)
        self.A_out = A_out
        return A_out

    def diff(self, epsilon, dJ_dA_out, lambda_=0., op_count=0, last_layer=False):
        if (
            self.W is None
            or self.B is None
            or self.A_in is None
            or self.A_out is None
        ):
            return None

        # Compute W derivatives
        (n, u) = self.W.shape  # n is number of parameters of W, u is number of units in the layer
        (m, _) = self.A_in.shape  # m is the number of samples used in the training set.
        self.dJ_dW_prev = copy.deepcopy(self.dJ_dW)
        self.dJ_dW = np.zeros_like(self.W)
        dA_out = np.zeros((m, u))
        # Loop through every W parameter and calculate dJ_jW for that parameter
        # Can do all units at once for each parameter.
        Epsilon = np.zeros((m, u))
        Epsilon_base = self.A_in * epsilon
        Z_base = np.matmul(self.A_in, self.W) + self.B
        for i in range(n):
            for j in range(u):
                op_count += 1
                dA_out.fill(0.)
                Epsilon.fill(0.)
                Epsilon[:, j] = Epsilon_base[:, i]
                Z_diff = Z_base + Epsilon
                A_out_diff = self.activation(Z_diff)
                dA_out = A_out_diff - self.A_out
                self.dJ_dW[i, j] = np.sum((dA_out / epsilon) * dJ_dA_out) / m + lambda_ / m * self.W[i, j]

        # Compute B derivatives
        self.dJ_dB_prev = copy.deepcopy(self.dJ_dB)
        self.dJ_dB = np.zeros_like(self.B)
        Epsilon = np.zeros_like(self.B)
        for j in range(u):
            op_count += 1
            dA_out.fill(0.)
            Epsilon.fill(0.)
            Epsilon[j] = epsilon
            B_diff = self.B + Epsilon
            Z_diff = np.matmul(self.A_in, self.W) + B_diff
            A_out_diff = self.activation(Z_diff)
            dA_out = A_out_diff - self.A_out
            self.dJ_dB[j] = np.sum((dA_out / epsilon) * dJ_dA_out) / m

        dJ_dA_in = np.zeros_like(self.A_in)
        if not last_layer:
            # Compute A_in derivatives
            Epsilon = np.zeros_like(self.A_in)
            A_out_base = self.activation(np.matmul(self.A_in, self.W) + self.B)
            A_out_diffs = np.zeros((m, u, n))
            for j in range(n):
                Epsilon.fill(0.)
                Epsilon[:, j] = epsilon
                A_in_diff = self.A_in + Epsilon
                A_out_diffs[:, :, j] = self.activation(np.matmul(A_in_diff, self.W) + self.B)

            for i in range(m):
                for j in range(n):
                    op_count += 1
                    A_out_diff = copy.deepcopy(A_out_base)
                    A_out_diff[i, :] = A_out_diffs[i, :, j]
                    dA_out = A_out_diff - self.A_out
                    dJ_dA_in[i, j] = np.sum((dA_out / epsilon) * dJ_dA_out) / m

        return (dJ_dA_in, op_count)

    def init_alpha(self, alpha):
        self.alpha = alpha
        self.alpha_W = np.ones_like(self.dJ_dW) * alpha
        self.alpha_B = np.ones_like(self.dJ_dB) * alpha

    def update_W_B(self, adam=None, lambda_=None):

        if adam is not None:
            adam_W = np.ones_like(self.dJ_dW)

            if self.dJ_dW_prev is None:
                dir_change_W = np.zeros_like(self.dJ_dW).astype(bool)
            else:
                dir_change_W = ((self.dJ_dW * self.dJ_dW_prev) < 0.0)

            if self.dir_change_W_prev is None:
                self.dir_change_W_prev = dir_change_W

            adam_W = (
                adam_W
                - dir_change_W * adam
                + (np.ones_like(dir_change_W).astype(float) - dir_change_W.astype(float)) * adam
            )
            self.alpha_W = self.alpha_W * adam_W

            self.dir_change_W_prev = dir_change_W

            adam_B = np.ones_like(self.dJ_dB)
            if self.dJ_dB_prev is None:
                dir_change_B = np.zeros_like(self.dJ_dB).astype(bool)
            else:
                dir_change_B = ((self.dJ_dB * self.dJ_dB_prev) < 0.0)

            if self.dir_change_B_prev is None:
                self.dir_change_B_prev = dir_change_B

            adam_B = (
                adam_B
                - dir_change_B * adam
                + (np.ones_like(dir_change_B).astype(float) - dir_change_B.astype(float)) * adam
            )
            self.alpha_B = self.alpha_B * adam_B

            self.dir_change_B_prev = dir_change_B

        self.W = self.W - self.alpha_W * self.dJ_dW
        self.B = self.B - self.alpha_B * self.dJ_dB

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(np.zeros_like(z), z)

    def linear(self, z):
        return z

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def set_W_B(self, W_new, B_new):
        self.W = W_new
        self.B = B_new


class Sequential():
    layers = None
    cost_function = None

    def __init__(self, layers, X=None, cost_function="mean_squared"):
        self.layers = layers

        if X is not None:
            self.init_W_B_from_X(X)

        if cost_function == "mean_squared":
            self.cost_function = self.mean_squared
        if cost_function == "binary_crossentropy":
            self.cost_function = self.binary_crossentropy
        else:
            self.cost_function = self.mean_squared

    def mean_squared(self, Y_hat, Y):
        (m, n) = Y_hat.shape
        L = 0.5 * np.power((Y_hat - Y), 2)
        cost = np.sum(L, 0) / m
        return (cost, L)

    def binary_crossentropy(self, Y_hat, Y):
        (m, n) = Y_hat.shape
        ones = np.ones_like(Y)
        L = -Y * np.log(np.clip(Y_hat, 1e-10, 1e100)) - (ones - Y) * np.log(np.clip((ones - Y_hat), 1e-10, 1e100))
        cost = np.sum(L, 0) / m
        return (cost, L)

    def init_layers(self, X: np.array, alpha: float):
        if self.layers is None:
            print("ERROR: Layers not initialized.")
            return
        if len(X.shape) > 2:
            print("ERROR: X must be a 2-D array.")
            return

        input_count = X.shape[-1]

        layer_count = 0
        for layer in self.layers:
            if layer.name is None:
                layer.name = f"layer{layer_count}"
            layer_count += 1
            layer.init_alpha(alpha)
            layer.init_W_B(input_count)
            input_count = layer.units

    def propagate(self, X):
        A_out = X
        # print(f"A_out: {A_out}")
        for layer in self.layers:
            A_in = A_out
            A_out = layer.propagate(A_in)

        return A_out

    def train(self, X, Y, alpha, iterations=10, epsilon=1e-6, adam=None, lambda_=0.):
        self.init_layers(X, alpha)

        J_hist = []

        (m, n) = X.shape
        for i in range(iterations):
            Y_hat = self.propagate(X)
            (J, Loss_Y_hat) = self.cost_function(Y_hat, Y)
            J_hist.append(J)

            Y_hat_test = copy.deepcopy(Y_hat)
            Y_hat_test[0, 0] += epsilon
            (J_test, Loss_Y_hat_test) = self.cost_function(Y_hat_test, Y)

            op_count = 0
            # Compute dJ_dY_hat
            dJ_dY_hat = np.zeros_like(Y_hat)
            for j in range(m):

                op_count += 1
                Epsilon = np.zeros_like(Y_hat)
                Epsilon[j, 0] = epsilon
                Y_hat_diff = Y_hat + Epsilon
                (J_diff, _) = self.cost_function(Y_hat_diff, Y)

                dJ = J_diff - J
                dJ_dY_hat[j] = dJ / epsilon

            dJ_dA_out = dJ_dY_hat

            layer_count = 0
            for layer in reversed(self.layers):
                layer_count += 1
                (dJ_dA_out, op_count) = layer.diff(
                    epsilon,
                    dJ_dA_out,
                    lambda_=lambda_,
                    op_count=op_count,
                    last_layer=(layer_count == len(self.layers))
                )

            for layer in self.layers:
                layer.update_W_B(adam=adam, lambda_=lambda_)

            if i % math.ceil(iterations / 10) == 0:
                print(f"Epoch {i}: Cost = {J}")

            if J < 1e-6:
                print(f"J threshold reached. Breaking training loop...")
                break

        print(f"J_initial: {J_hist[0]}   J_final: {J_hist[-1]}")


def normalize(X):
    (m, n) = X.shape
    means = np.mean(X, axis=0)
    stds = np.clip(np.std(X, axis=0), 1, 1e100)

    return (X - means) / stds


def map_features(X1, X2, order):
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    out = []
    for i in range(1, order + 1):
        # print(X1_feature)
        for j in range(i + 1):
            new_feature = (X1 ** (i - j)) * (X2 ** j)
            out.append(new_feature)
    final_out = np.stack(out, axis=1)
    return final_out


# X, Y = load_coffee_data()

X = np.load("./X_coffee.npy")
Y = np.load("./Y_coffee.npy")

mapping_order = 8
epochs = 100
alpha = 1e-3
alpha_tf = 1e-3
adam = 0.00
lambda_ = 0.000000
epsilon = 1e-6

Xn = normalize(X)
# Add silly point
# Xn[-1] = np.array([0.365, 0.410])
# Xn[0] = np.array([-0.83455487, -0.65287939])
Xn_mapped = map_features(Xn[:, 0], Xn[:, 1], mapping_order)
# Y[-1] = 1

# X_tst = np.array([
#     [200, 13.9],  # postive example
#     [200, 17]])   # negative example
# X_tstn = normalize(X_tst)  # remember to normalize
# X_tstn_mapped = map_features(X_tstn[:, 0], X_tstn[:, 1], mapping_order)

X_tst = X[0:1, :]
X_tstn = normalize(X_tst)  # remember to normalize
X_tstn_mapped = map_features(X_tstn[:, 0], X_tstn[:, 1], mapping_order)

model = Sequential(
    [
        Layer(units=25, activation="relu"),
        Layer(units=15, activation="relu"),
        Layer(units=1, activation="sigmoid")
    ],
    cost_function="binary_crossentropy"
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
model.train(Xn_mapped, Y, alpha, iterations=epochs, epsilon=epsilon, adam=adam, lambda_=lambda_)
end = datetime.now()
print(f"=========================================================")
output = (model.propagate(X_tstn_mapped) >= 0.5).astype(int)

fig, ax = plt.subplots()
ones_filter = (Y > 0.5)
X_ones = Xn[ones_filter[:, 0]]
ax.scatter(X_ones[:, 0], X_ones[:, 1], marker='o', facecolor='white', edgecolor='blue')
zeros_filter = (Y < 0.5)
X_zeros = Xn[zeros_filter[:, 0]]
ax.scatter(X_zeros[:, 0], X_zeros[:, 1], marker='x', facecolor='red')


x0_boundary = np.linspace(np.min(Xn[:, 0]), np.max(Xn[:, 0]), 50)
x1_boundary = np.linspace(np.min(Xn[:, 1]), np.max(Xn[:, 1]), 50)
X0_boundary, X1_boundary = np.meshgrid(x0_boundary, x1_boundary)
Z_boundary = np.zeros_like(X0_boundary)
for i in range(x0_boundary.size):
    for j in range(x1_boundary.size):
        X_boundary_mapped = map_features(X0_boundary[i, j], X1_boundary[i, j], mapping_order)
        z = model.propagate(X_boundary_mapped)
        Z_boundary[i, j] = z[0, 0]

ax.contour(X0_boundary, X1_boundary, Z_boundary, levels=[0.5], colors="g")
plt.show()
