import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from training_data import X_train, y_train


def sigmoid(z):
    # z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def cost(X, y, w, b, lambda_=0.):
    m = X.shape[0]

    cost = 0.
    regularization = 0.

    for i in range(m):
        z = np.dot(X[i, :], w) + b
        if y[i] == 1:
            to_log = sigmoid(z)
        else:
            to_log = 1 - sigmoid(z)

        if to_log <= 0:
            to_log = 1e-100
        cost += -np.log(to_log)
    regularization += np.sum(w ** 2)

        # print(f"i: {i}  X: {X[i]}  y: {y[i]}  sigmoid(z): {sigmoid(z)}    to_log: {to_log}    cost_added: {-np.log(to_log)}\n")

    return (cost / m) + (regularization * lambda_ / (2. * m))


def regression(X, y, w, b, lambda_=0.):

    dj_dw = np.zeros_like(w)
    dj_db = 0.
    (m, n) = X.shape

    z = np.c_[np.dot(X, w) + b]
    for j in range(n):
        dj_dw[j] = np.sum((sigmoid(z) - y) * np.c_[X[:, j]]) / m + lambda_ / m * w[j]

    gradient_sum = 0
    gradient_sum = np.sum(sigmoid(z) - y)
    dj_db = gradient_sum / m

    return (dj_dw, dj_db)


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_=0.):
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    for i in range(num_iters):
        (dj_dw, db_dw) = regression(X, y, w, b, lambda_=lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * db_dw
        new_cost = cost(X, y, w, b)
        J_history.append(new_cost)

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return (w, b, J_history)


def plot_logistic_data(X, y, w, b):
    (m, n) = X.shape
    ones_count = np.sum(y)
    zeros_count = m - ones_count

    data_ones = np.zeros((ones_count, n))
    data_zeros = np.zeros((zeros_count, n))

    ones_index = 0
    zeros_index = 0

    for i in range(m):
        if y[i, 0] == 1:
            data_ones[ones_index, :] = X[i, :]
            ones_index += 1
        else:
            data_zeros[zeros_index, :] = X[i, :]
            zeros_index += 1

    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])

    y_min = -1 * (x_min * w[0] + b) / w[1]
    y_max = -1 * (x_max * w[0] + b) / w[1]

    fig, ax = plt.subplots()
    ax.scatter(data_ones[:, 0], data_ones[:, 1], marker="x", c="red", label="1")
    ax.scatter(data_zeros[:, 0], data_zeros[:, 1], marker="o", facecolors='none', edgecolors='blue', label="0")
    ax.plot([x_min, x_max], [y_min, y_max])
    ax.legend(loc="best")
    plt.show()


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


def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y[:, 0] == 1
    negative = y[:, 0] == 0

    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)


def plot_decision_boundary(w, b, X, y):
    # Credit to dibgerge on Github for this plotting code

    plot_data(X[:, 0:2], y)

    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)

        plt.plot(plot_x, plot_y, c="b")

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))

        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid(np.dot(map_features(u[i], v[j], 6), w) + b)

        # important to transpose z before calling contour
        z = z.T

        # Plot z = 0.5
        plt.contour(u, v, z, levels=[0.5], colors="g")


X_mapped = map_features(X_train[:, 0], X_train[:, 1], 6)

# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01

# Some gradient descent settings
iterations = 10000
alpha = 0.01

(w, b, J_history) = gradient_descent(X_mapped, y_train, initial_w, initial_b, alpha, iterations, lambda_)

plot_decision_boundary(w, b, X_mapped, y_train)
# Set the y-axis label
plt.ylabel('Microchip Test 2')
# Set the x-axis label
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

# w_init = np.zeros_like(np.c_[X_train[0]])
# w_init = np.array([[0], [0]])
# b_init = -0.
# alpha = 0.001
# iters = 1000

# w_out, b_out, J_hist = gradient_descent(X_train, y_train, w_init, b_init, alpha, iters)
# print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
# print(f"final w: {w_out}    final b: {b_out}")
# # print(f"J_hist: {J_hist}")

# fig, ax = plt.subplots()
# ax.plot(J_hist)
# plt.show()

plot_logistic_data(X_train, y_train, w, b)
