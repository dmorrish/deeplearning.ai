import numpy as np
import math
import copy


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
            print("WARNING: W and B were already initialized.")
            return
        if self.units is None:
            print("ERROR: Unit count was never initialized.")
            return
        if num_inputs < 1:
            print("ERROR: The layer must have at least one input.")
            return
        rng = np.random.default_rng(2)
        self.W = rng.random((num_inputs, self.units)) * 1e-3
        self.B = rng.random((self.units,)) * 1e-3

    def propagate(self, A_in):
        self.A_in = A_in
        Z = np.matmul(A_in, self.W) + self.B
        A_out = self.activation(Z)
        self.A_out = A_out
        return A_out

    def diff(self, epsilon, dJ_dA_out, lambda_=0., last_layer=False):
        if (
            self.W is None
            or self.B is None
            or self.A_in is None
            or self.A_out is None
        ):
            return None

        # Compute W derivatives more efficiently
        (n, u) = self.W.shape  # n is number of parameters of W, u is number of units in the layer
        (m, _) = self.A_in.shape  # m is the number of samples used in the training set.
        self.dJ_dW_prev = copy.deepcopy(self.dJ_dW)
        self.dJ_dW = np.zeros_like(self.W)
        # Create array of dA_out_dWs for each parameter number being changed
        Epsilon = np.zeros_like(self.W)
        dA_out_dWs = np.zeros((m, u, n))
        for j in range(n):
            Epsilon[j, :].fill(epsilon)
            W_diff = self.W + Epsilon
            Z_diff = np.matmul(self.A_in, W_diff) + self.B
            A_out_diff = self.activation(Z_diff)
            dA_out_dWs[:, :, j] = (A_out_diff - self.A_out) / epsilon
            Epsilon[j, :].fill(0.0)
        # Loop through every W parameter and calculate dJ_jW for that parameter
        # Can do all units at once for each parameter.
        for i in range(n):
            for j in range(u):
                dA_out_dW = np.zeros((m, u))
                dA_out_dW[:, j] = dA_out_dWs[:, j, i]
                self.dJ_dW[i, j] = np.sum(dA_out_dW * dJ_dA_out) + lambda_ / m * self.W[i, j]

        # Compute B derivatives
        self.dJ_dB_prev = copy.deepcopy(self.dJ_dB)
        self.dJ_dB = np.zeros_like(self.B)
        Epsilon = np.zeros_like(self.B)
        for j in range(u):
            Epsilon.fill(0.)
            Epsilon[j] = epsilon
            B_diff = self.B + Epsilon
            Z_diff = np.matmul(self.A_in, self.W) + B_diff
            A_out_diff = self.activation(Z_diff)
            dA_out = A_out_diff - self.A_out
            self.dJ_dB[j] = np.sum((dA_out / epsilon) * dJ_dA_out)

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

            A_out_diff = copy.deepcopy(A_out_base)
            for i in range(m):
                for j in range(n):
                    # A_out_diff = copy.deepcopy(A_out_base)
                    A_out_diff[i, :] = A_out_diffs[i, :, j]
                    dA_out = A_out_diff - self.A_out
                    dJ_dA_in[i, j] = np.sum((dA_out / epsilon) * dJ_dA_out)
                    A_out_diff[i, :] = A_out_base[i, :]

        return dJ_dA_in

    def init_alpha(self, alpha):
        self.alpha = alpha
        self.alpha_W = np.ones_like(self.dJ_dW) * alpha
        self.alpha_B = np.ones_like(self.dJ_dB) * alpha

    def update_W_B(self, adam=None, lambda_=None):

        if adam is not None and adam > 0:
            if self.dJ_dW_prev is None:
                dir_change_W = np.zeros_like(self.dJ_dW).astype(bool)
            else:
                dir_change_W = ((self.dJ_dW * self.dJ_dW_prev) < 0.0)

            if self.dir_change_W_prev is None:
                self.dir_change_W_prev = dir_change_W

            adam_W = (
                np.ones_like(self.dJ_dW)
                - dir_change_W * adam
                - (self.dJ_dW >= 0.0).astype(float) * adam
                + (self.dJ_dW < 0.0).astype(float) * adam
            )
            self.alpha_W = self.alpha_W * adam_W

            self.dir_change_W_prev = dir_change_W

            if self.dJ_dB_prev is None:
                dir_change_B = np.zeros_like(self.dJ_dB).astype(bool)
            else:
                dir_change_B = ((self.dJ_dB * self.dJ_dB_prev) < 0.0)

            if self.dir_change_B_prev is None:
                self.dir_change_B_prev = dir_change_B

            adam_B = (
                np.ones_like(self.dJ_dB)
                - dir_change_B * adam
                - (self.dJ_dB >= 0.0).astype(float) * adam
                + (self.dJ_dB < 0.0).astype(float) * adam
            )
            self.alpha_B = self.alpha_B * adam_B

            self.dir_change_B_prev = dir_change_B

        self.W = self.W - self.dJ_dW * self.alpha_W
        self.B = self.B - self.dJ_dB * self.alpha_B

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
    from_logits = False

    def __init__(self, layers, X=None, cost_function=None, from_logits=False):
        self.layers = layers

        if X is not None:
            self.init_W_B_from_X(X)

        if cost_function == "mean_squared":
            self.cost_function = self.mean_squared
        elif cost_function == "binary_crossentropy":
            self.cost_function = self.binary_crossentropy
        elif cost_function == "sparse_categorical_crossentropy":
            self.cost_function = self.sparse_categorical_crossentropy
        else:
            self.cost_function = self.mean_squared

        self.from_logits = from_logits

    def softmax(self, z):
        (m, n) = z.shape
        z = np.clip(z, -500, 500)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1).reshape(m, 1)

    def mean_squared(self, Y_hat, Y):
        (m, n) = Y_hat.shape
        L = 0.5 * np.power((Y_hat - Y), 2)
        cost = np.sum(L, 0) / m
        return (cost, L)

    def binary_crossentropy(self, Y_hat, Y):
        (m, n) = Y_hat.shape
        ones = np.ones_like(Y)
        L = -Y * np.log(np.clip(Y_hat, 1e-100, 1e100)) - (ones - Y) * np.log(np.clip((ones - Y_hat), 1e-100, 1e100))
        cost = np.sum(L, 0) / m
        return (cost, L)

    def sparse_categorical_crossentropy(self, Y_hat, Y):
        (m, u) = Y_hat.shape
        L = np.zeros((m,)).astype(float)
        if self.from_logits is True:
            pool = self.softmax(Y_hat)
            for i in range(m):
                to_log = np.clip(pool[i, Y[i, 0].astype(int)], 1e-100, 1e100)
                L[i] = -np.log(to_log)
        else:
            for i in range(len(Y)):
                L[i] = -np.log(np.clip(Y_hat[i, Y[i]], 1e-100, 1e100))

        cost = np.sum(L) / m
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
        for layer in self.layers:
            A_in = A_out
            A_out = layer.propagate(A_in)

        return A_out

    def train(self, X, Y, alpha, iterations=10, epsilon=1e-6, adam=None, lambda_=0., verbose=False):
        self.init_layers(X, alpha)

        if verbose is True:
            print("==============")
            print("Initial State:")
            print("==============\n")
            for layer in self.layers:
                print(f"{layer.name}:")
                print(f"\nW:\n{csv_1D_2D_array(layer.W)}")
                print(f"\nB:\n{csv_1D_2D_array(layer.B)}")
                print(f"\ndJ_dW:\n{csv_1D_2D_array(layer.dJ_dW)}")
                print(f"\ndJ_dB:\n{csv_1D_2D_array(layer.dJ_dB)}")

        J_hist = []

        (m, n) = X.shape
        outs = self.layers[-1].units
        for epoch in range(iterations):
            Y_hat = self.propagate(X)
            (J, Loss_Y_hat) = self.cost_function(Y_hat, Y)
            J_hist.append(J)

            # Compute dJ_dY_hat
            dJ_dY_hat = np.zeros_like(Y_hat)
            J_diff = np.zeros_like(Y_hat)
            Loss_Y_hat_diffs = np.zeros((m, outs))
            for j in range(outs):
                Epsilon = np.zeros_like(Y_hat)
                Epsilon[:, j] = epsilon
                Y_hat_diff = Y_hat + Epsilon
                (_, Loss_Y_hat_diffs[:, j]) = self.cost_function(Y_hat_diff, Y)

            for i in range(m):
                for j in range(outs):
                    J_diff[i, j] = J - Loss_Y_hat[i] / m + Loss_Y_hat_diffs[i, j] / m

            dJ_dY_hat = (J_diff - J) / epsilon
            dJ_dA_out = dJ_dY_hat

            layer_count = 0
            for layer in reversed(self.layers):
                layer_count += 1
                dJ_dA_out = layer.diff(
                    epsilon,
                    dJ_dA_out,
                    lambda_=lambda_,
                    last_layer=(layer_count == len(self.layers))
                )

            for layer in self.layers:
                layer.update_W_B(adam=adam, lambda_=lambda_)

            if verbose is True:
                print("\n========================")
                print(f"State after iteration {epoch + 1}:")
                print("========================\n")
                print(f"Y_hat:\n{Y_hat}")
                print(f"dJ_dY_hat:\n{dJ_dY_hat}")
                for layer in self.layers:
                    print(f"\n{layer.name} W (new):\n{csv_1D_2D_array(layer.W)}")
                    print(f"\n{layer.name} B (new):\n{csv_1D_2D_array(layer.B)}")
                    print(f"\n{layer.name} dJ_dW:\n{csv_1D_2D_array(layer.dJ_dW)}")
                    print(f"\n{layer.name} dJ_dB:\n{csv_1D_2D_array(layer.dJ_dB)}")
                    print(f"\n{layer.name} alpha_W:\n{layer.alpha_W}")
                    print(f"\n{layer.name} alpha_B:\n{layer.alpha_B}")
                    print(f"\n{layer.name} A_out:\n{csv_1D_2D_array(layer.A_out)}")

            if epoch % math.ceil(iterations / 10) == 0:
                print(f"Epoch {epoch}: Cost = {J}")

            # if J < 1e-6:
            #     print(f"J threshold reached. Breaking training loop...")
            #     break

        print(f"J_initial: {J_hist[0]}   J_final: {J_hist[-1]}")


def csv_1D_2D_array(X):
    if X is None:
        return "None"
    if len(X.shape) != 2 and len(X.shape) != 1:
        print("YOU CAN ONLY PRINT 1D vectors or 2D arrays.")
        return

    print_val = ""
    if len(X.shape) == 1:
        (m, ) = X.shape
        for i in range(m):
            print_val += f"{X[i]}, "
        print_val = print_val[0:-2]
    else:
        (m, n) = X.shape
        for i in range(m):
            for j in range(n):
                print_val += f"{X[i, j]}, "
            print_val = print_val[0:-2]
            print_val += "\n"
        print_val = print_val[0:-1]

    return print_val
