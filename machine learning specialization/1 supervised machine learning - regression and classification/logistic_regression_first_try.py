import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(X, y, w, b):
    # 1 / 2m * sum((f(x) - y) ^ 2)
    m = X.shape[0]
    z = np.dot(X, w) + b
    error = sigmoid(z) - y
    cost = np.sum(error ** 2) / (2 * m)
    return cost


def regression(X, y, w, b):

    dj_dw = np.zeros_like(w)
    (m, n) = X.shape

    f_x = np.dot(X, w) + b
    # print(f"f_x: {f_x}")
    inner_calc = (sigmoid(f_x) - y) * (sigmoid(f_x) ** 2) * np.exp(-f_x)

    for i in range(n):
        # repeat once for every parameter in X
        to_sum = inner_calc * np.c_[X[:, i]]
        # print(f"inner_calc: {inner_calc}")
        # print(f"X[:, {i}]: {np.c_[X[:, i]]}")
        # print(f"to_sum: {to_sum}")
        dj_dwn = np.sum(to_sum) / m
        dj_dw[i, 0] = dj_dwn

    to_sum = inner_calc
    db_dw = np.sum(to_sum) / m
    return (dj_dw, db_dw)


X_train = np.array([
    [0.20, 2.60],
    [0.90, 0.70],
    [0.20, 2.20],
    [0.10, 2.60],
    [0.10, 2.60],
    [0.50, 0.80],
    [1.00, 0.10],
    [0.90, 1.50],
    [0.10, 2.90],
    [0.90, 0.50],
    [0.90, 0.60],
    [0.40, 2.50],
    [0.60, 0.80],
    [0.10, 1.90],
    [0.70, 1.40],
    [0.20, 0.90],
    [0.20, 0.80],
    [0.60, 1.80],
    [0.70, 1.50],
    [0.60, 2.20],
    [1.50, 0.60],
    [0.80, 0.30],
    [0.50, 0.70],
    [0.70, 0.40],
    [1.00, 0.00],
    [0.60, 0.90],
    [1.20, 1.20],
    [0.70, 1.80],
    [1.00, 0.50],
    [0.90, 1.40],
    [0.50, 1.80],
    [1.10, 1.20],
    [0.30, 1.50],
    [1.40, 0.90],
    [0.30, 2.00],
    [0.70, 1.20],
    [0.00, 4.70],
    [0.40, 4.20],
    [0.70, 5.00],
    [0.20, 4.70],
    [1.20, 2.90],
    [0.10, 5.80],
    [1.30, 4.20],
    [1.10, 4.30],
    [1.20, 4.40],
    [0.40, 4.90],
    [1.40, 3.80],
    [1.40, 3.70],
    [1.20, 4.80],
    [1.00, 4.90],
    [0.00, 5.40],
    [0.90, 4.20],
    [1.30, 3.40],
    [0.80, 5.00],
    [0.00, 5.60],
    [0.00, 4.60],
    [1.50, 4.40],
    [0.90, 4.20],
    [1.10, 3.00],
    [0.20, 4.30],
    [0.90, 3.80],
    [0.40, 5.30],
    [0.80, 3.60],
    [1.30, 2.70],
    [1.00, 3.50],
    [1.40, 4.20],
    [0.90, 5.00],
    [0.20, 5.80],
    [0.80, 4.20],
    [1.20, 3.60],
    [1.00, 3.60],
    [0.50, 4.50]
])

y_train = np.array([
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
    [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1],
    [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
    [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
    [1], [1]
])


w = np.array([[-100], [10]])
b = 0

alpha = 1e2
num_iters = 1000

for i in range(num_iters):
    (dj_dw, db_dw) = regression(X_train, y_train, w, b)
    # print(f"dj_dw: {dj_dw}  db_dw: {db_dw}")
    # print(f"w: {w}   alpha: {alpha}")
    w = w - alpha * dj_dw
    b = b - alpha * db_dw
    new_cost = cost(X_train, y_train, w, b)

    if i % 100 == 0:
        print(f"Iteration {i}: cost = {new_cost}")

print(f"final w: {w}    final b: {b}")

X_test = np.array([0.6, 1.926])

z = np.dot(X_test, w) + b
probability = sigmoid(z)

print(f"Probability for point {X_test}: {probability}")
