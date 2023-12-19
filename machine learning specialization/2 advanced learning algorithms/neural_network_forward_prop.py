import numpy as np
import tensorflow as tf


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

    def __init__(self, units: int = 1, activation: str = "linear"):
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

    def init_W_B(self, num_inputs: int):
        if self.W is not None or self.B is not None:
            print("ERROR: W and B were already initialized.")
            return
        if self.units is None:
            print("ERROR: Unit count was never initialized.")
        if num_inputs < 1:
            print("ERROR: The layer must have at least one input.")
            return
        rng = np.random.default_rng()
        self.W = rng.random((num_inputs, self.units))
        print(f"W:\n{self.W}")
        self.B = rng.random((self.units,))
        print(f"B:\n{self.B}")

    def propagate(self, A_in):
        Z = np.matmul(A_in, self.W) + self.B
        print(f"Z:\n{Z}")
        A = self.activation(Z)
        print(f"A:\n{A}")
        return A

    def sigmoid(self, z):
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

    def __init__(self, layers, X=None):
        self.layers = layers

        if X is not None:
            self.init_W_B_from_X(X)

    def init_W_B_from_X(self, X: np.array):
        if self.layers is None:
            print("ERROR: Layers not initialized.")
            return
        if len(X.shape) > 2:
            print("ERROR: X must be a 2-D array.")
            return

        input_count = X.shape[-1]

        for layer in self.layers:
            layer.init_W_B(input_count)
            input_count = layer.units

    def propagate(self, X):
        a_out = X
        for layer in self.layers:
            a_in = a_out
            a_out = layer.propagate(a_in)
        return a_out


layer_test = Layer(4, activation="softmax")
A_test = np.array([1., 2., 2.7, 7.3])
Z_test = layer_test.softmax(A_test)
print(f"A:\n{A_test}")
print(f"Z:\n{Z_test}")
exit(1)

X, Y = load_coffee_data()
print(f"X and y shapes: {X.shape} and {Y.shape}")

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

X_tst = np.array([
    [200, 13.9],  # postive example
    [200, 17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize

model = Sequential([
    Layer(units=3, activation="sigmoid"),
    Layer(units=1, activation="sigmoid")
])

model.init_W_B_from_X(X)
model.layers[0].set_W_B(np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]]), np.array([[-9.82, -9.28, 0.96]]))
model.layers[1].set_W_B(np.array([[-31.18], [-27.59], [-32.56]]), np.array([[15.41]]))

output = (model.propagate(X_tstn) >= 0.5).astype(int)
print(f"final output:\n{output}")
