import numpy as np
import math
import copy


def sigmoid(z):
    # print(f"sigmoid result: {1 / (1 + np.exp(-z))}")
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def cost(X, y, w, b):
    # 1 / 2m * sum((f(x) - y) ^ 2)
    m = X.shape[0]

    cost = 0

    for i in range(m):
        z = np.dot(X[i, :], w) + b
        if y[i] == 1:
            to_log = sigmoid(z)
            # if to_log <= 0:
            #     to_log = 1e-30
            cost += -np.log(to_log)
        else:
            # print(f"z: {z}")
            # print(f"sigmoid(x): {sigmoid(z)}")
            to_log = 1 - sigmoid(z)
            # if to_log <= 0:
            #     to_log = 1e-30
            cost += -np.log(to_log)

    return cost / m


def regression(X, y, w, b):

    dj_dw = np.zeros_like(w)
    (m, n) = X.shape

    for j in range(n):
        gradient_sum = 0

        z = np.dot(X, w) + b
        gradient_sum = np.sum((sigmoid(z) - y) * np.c_[X[:, j]])

        dj_dwn = gradient_sum / m
        dj_dw[j, 0] = dj_dwn

    gradient_sum = 0
    gradient_sum = np.sum(sigmoid(z) - y)
    db_dw = gradient_sum / m

    return (dj_dw, db_dw)


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    for i in range(num_iters):
        # alpha = alpha_base * math.pow(2, int(i / 1000))
        (dj_dw, db_dw) = regression(X, y, w, b)
        # print(f"dj_dw: {dj_dw}  db_dw: {db_dw}")
        # print(f"w: {w}   alpha: {alpha}")
        w = w - alpha * dj_dw
        b = b - alpha * db_dw
        new_cost = cost(X, y, w, b)
        J_history.append(new_cost)

        # if i % 1000 == 0:
        #     print(f"Iteration {i}: cost = {new_cost}")
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")

    return (w, b, J_history)


X_train = np.array([
    [0.80, 2.60],
    [3.60, 0.70],
    [0.80, 2.20],
    [0.40, 2.60],
    [0.40, 2.60],
    [2.00, 0.80],
    [4.00, 0.10],
    [3.60, 1.50],
    [0.40, 2.90],
    [3.60, 0.50],
    [3.60, 0.60],
    [1.60, 2.50],
    [2.40, 0.80],
    [0.40, 1.90],
    [2.80, 1.40],
    [0.80, 0.90],
    [0.80, 0.80],
    [2.40, 1.80],
    [2.80, 1.50],
    [2.40, 2.20],
    [6.00, 0.60],
    [3.20, 0.30],
    [2.00, 0.70],
    [2.80, 0.40],
    [4.00, 0.00],
    [2.40, 0.90],
    [4.80, 1.20],
    [2.80, 1.80],
    [4.00, 0.50],
    [3.60, 1.40],
    [2.00, 1.80],
    [4.40, 1.20],
    [1.20, 1.50],
    [5.60, 0.90],
    [1.20, 2.00],
    [2.80, 1.20],
    [0.00, 4.70],
    [1.60, 4.20],
    [2.80, 5.00],
    [0.80, 4.70],
    [4.80, 2.90],
    [0.40, 5.80],
    [5.20, 4.20],
    [4.40, 4.30],
    [4.80, 4.40],
    [1.60, 4.90],
    [5.60, 3.80],
    [5.60, 3.70],
    [4.80, 4.80],
    [4.00, 4.90],
    [0.00, 5.40],
    [3.60, 4.20],
    [5.20, 3.40],
    [3.20, 5.00],
    [0.00, 5.60],
    [0.00, 4.60],
    [6.00, 4.40],
    [3.60, 4.20],
    [4.40, 3.00],
    [0.80, 4.30],
    [3.60, 3.80],
    [1.60, 5.30],
    [3.20, 3.60],
    [5.20, 2.70],
    [4.00, 3.50],
    [5.60, 4.20],
    [3.60, 5.00],
    [0.80, 5.80],
    [3.20, 4.20],
    [4.80, 3.60],
    [4.00, 3.60],
    [2.00, 4.50]
])

y_train = np.array([
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
    [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1],
    [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
    [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1],
    [1], [1]
])


X_tmp = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([[0], [0], [0], [1], [1], [1]])
w_tmp = np.array([[2.], [3.]])
b_tmp = 1.
dj_dw_tmp, dj_db_tmp = regression(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}")
print(f"dj_dw: {dj_dw_tmp.tolist()}")

w_tmp = np.zeros_like(np.c_[X_train[0]])

b_tmp = 0.
alpha = 0.1
iters = 10000

print(X_tmp)
print(y_tmp)

w_out, b_out, _ = gradient_descent(X_tmp, y_tmp, w_tmp, b_tmp, alpha, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0, 0, 0, 1, 1, 1])


exit(1)


w = np.array([[0], [0]])
b = 0

alpha_base = 3e1
num_iters = 50000



print(f"final w: {w}    final b: {b}")

X_test = np.array([0.6, 1.926])

z = np.dot(X_test, w) + b
z = z[0]
probability = sigmoid(z)

print(f"Probability for point {X_test}: {probability}")
