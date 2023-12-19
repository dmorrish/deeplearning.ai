import numpy as np
import matplotlib.pyplot as plt
import math


def gradient_descent_iteration(alpha, w, b, X, y):
    (m, n) = X.shape
    error = np.dot(X, w) + b - y
    dJ_dw = np.array([])
    for i in range(n):
        partial_derivative = np.sum(error * X[:, i]) / m
        dJ_dw = np.append(dJ_dw, partial_derivative)

    dJ_db = np.sum(error) / m

    w = w - alpha * dJ_dw
    b = b - alpha * dJ_db

    return (w, b)


def calc_cost(w, b, X, y):
    (m, n) = X.shape
    error = (np.dot(X, w) + b - y) ** 2
    cost = np.sum(error) / (2 * m)
    return cost


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = w_in
    b = b_in

    for i in range(num_iters):
        (w, b) = gradient_descent_iteration(alpha, w, b, X, y)
        J_history.append(calc_cost(w, b, X, y))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return (w, b, J_history)

# X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
# y_train = np.array([460, 232, 178])


X_train = np.array([
    [1.244e+03, 3.000e+00, 1.000e+00, 6.400e+01],
    [1.947e+03, 3.000e+00, 2.000e+00, 1.700e+01],
    [1.725e+03, 3.000e+00, 2.000e+00, 4.200e+01],
    [1.959e+03, 3.000e+00, 2.000e+00, 1.500e+01],
    [1.314e+03, 2.000e+00, 1.000e+00, 1.400e+01],
    [8.640e+02, 2.000e+00, 1.000e+00, 6.600e+01],
    [1.836e+03, 3.000e+00, 1.000e+00, 1.700e+01],
    [1.026e+03, 3.000e+00, 1.000e+00, 4.300e+01],
    [3.194e+03, 4.000e+00, 2.000e+00, 8.700e+01],
    [7.880e+02, 2.000e+00, 1.000e+00, 8.000e+01],
    [1.200e+03, 2.000e+00, 2.000e+00, 1.700e+01],
    [1.557e+03, 2.000e+00, 1.000e+00, 1.800e+01],
    [1.430e+03, 3.000e+00, 1.000e+00, 2.000e+01],
    [1.220e+03, 2.000e+00, 1.000e+00, 1.500e+01],
    [1.092e+03, 2.000e+00, 1.000e+00, 6.400e+01],
    [8.480e+02, 1.000e+00, 1.000e+00, 1.700e+01],
    [1.682e+03, 3.000e+00, 2.000e+00, 2.300e+01],
    [1.768e+03, 3.000e+00, 2.000e+00, 1.800e+01],
    [1.040e+03, 3.000e+00, 1.000e+00, 4.400e+01],
    [1.652e+03, 2.000e+00, 1.000e+00, 2.100e+01],
    [1.088e+03, 2.000e+00, 1.000e+00, 3.500e+01],
    [1.316e+03, 3.000e+00, 1.000e+00, 1.400e+01],
    [1.593e+03, 0.000e+00, 1.000e+00, 2.000e+01],
    [9.720e+02, 2.000e+00, 1.000e+00, 7.300e+01],
    [1.097e+03, 3.000e+00, 1.000e+00, 3.700e+01],
    [1.004e+03, 2.000e+00, 1.000e+00, 5.100e+01],
    [9.040e+02, 3.000e+00, 1.000e+00, 5.500e+01],
    [1.694e+03, 3.000e+00, 1.000e+00, 1.300e+01],
    [1.073e+03, 2.000e+00, 1.000e+00, 1.000e+02],
    [1.419e+03, 3.000e+00, 2.000e+00, 1.900e+01],
    [1.164e+03, 3.000e+00, 1.000e+00, 5.200e+01],
    [1.935e+03, 3.000e+00, 2.000e+00, 1.200e+01],
    [1.216e+03, 2.000e+00, 2.000e+00, 7.400e+01],
    [2.482e+03, 4.000e+00, 2.000e+00, 1.600e+01],
    [1.200e+03, 2.000e+00, 1.000e+00, 1.800e+01],
    [1.840e+03, 3.000e+00, 2.000e+00, 2.000e+01],
    [1.851e+03, 3.000e+00, 2.000e+00, 5.700e+01],
    [1.660e+03, 3.000e+00, 2.000e+00, 1.900e+01],
    [1.096e+03, 2.000e+00, 2.000e+00, 9.700e+01],
    [1.775e+03, 3.000e+00, 2.000e+00, 2.800e+01],
    [2.030e+03, 4.000e+00, 2.000e+00, 4.500e+01],
    [1.784e+03, 4.000e+00, 2.000e+00, 1.070e+02],
    [1.073e+03, 2.000e+00, 1.000e+00, 1.000e+02],
    [1.552e+03, 3.000e+00, 1.000e+00, 1.600e+01],
    [1.953e+03, 3.000e+00, 2.000e+00, 1.600e+01],
    [1.224e+03, 2.000e+00, 2.000e+00, 1.200e+01],
    [1.616e+03, 3.000e+00, 1.000e+00, 1.600e+01],
    [8.160e+02, 2.000e+00, 1.000e+00, 5.800e+01],
    [1.349e+03, 3.000e+00, 1.000e+00, 2.100e+01],
    [1.571e+03, 3.000e+00, 1.000e+00, 1.400e+01],
    [1.486e+03, 3.000e+00, 1.000e+00, 5.700e+01],
    [1.506e+03, 2.000e+00, 1.000e+00, 1.600e+01],
    [1.097e+03, 3.000e+00, 1.000e+00, 2.700e+01],
    [1.764e+03, 3.000e+00, 2.000e+00, 2.400e+01],
    [1.208e+03, 2.000e+00, 1.000e+00, 1.400e+01],
    [1.470e+03, 3.000e+00, 2.000e+00, 2.400e+01],
    [1.768e+03, 3.000e+00, 2.000e+00, 8.400e+01],
    [1.654e+03, 3.000e+00, 1.000e+00, 1.900e+01],
    [1.029e+03, 3.000e+00, 1.000e+00, 6.000e+01],
    [1.120e+03, 2.000e+00, 2.000e+00, 1.600e+01],
    [1.150e+03, 3.000e+00, 1.000e+00, 6.200e+01],
    [8.160e+02, 2.000e+00, 1.000e+00, 3.900e+01],
    [1.040e+03, 3.000e+00, 1.000e+00, 2.500e+01],
    [1.392e+03, 3.000e+00, 1.000e+00, 6.400e+01],
    [1.603e+03, 3.000e+00, 2.000e+00, 2.900e+01],
    [1.215e+03, 3.000e+00, 1.000e+00, 6.300e+01],
    [1.073e+03, 2.000e+00, 1.000e+00, 1.000e+02],
    [2.599e+03, 4.000e+00, 2.000e+00, 2.200e+01],
    [1.431e+03, 3.000e+00, 1.000e+00, 5.900e+01],
    [2.090e+03, 3.000e+00, 2.000e+00, 2.600e+01],
    [1.790e+03, 4.000e+00, 2.000e+00, 4.900e+01],
    [1.484e+03, 3.000e+00, 2.000e+00, 1.600e+01],
    [1.040e+03, 3.000e+00, 1.000e+00, 2.500e+01],
    [1.431e+03, 3.000e+00, 1.000e+00, 2.200e+01],
    [1.159e+03, 3.000e+00, 1.000e+00, 5.300e+01],
    [1.547e+03, 3.000e+00, 2.000e+00, 1.200e+01],
    [1.983e+03, 3.000e+00, 2.000e+00, 2.200e+01],
    [1.056e+03, 3.000e+00, 1.000e+00, 5.300e+01],
    [1.180e+03, 2.000e+00, 1.000e+00, 9.900e+01],
    [1.358e+03, 2.000e+00, 1.000e+00, 1.700e+01],
    [9.600e+02, 3.000e+00, 1.000e+00, 5.100e+01],
    [1.456e+03, 3.000e+00, 2.000e+00, 1.600e+01],
    [1.446e+03, 3.000e+00, 2.000e+00, 2.500e+01],
    [1.208e+03, 2.000e+00, 1.000e+00, 1.500e+01],
    [1.553e+03, 3.000e+00, 2.000e+00, 1.600e+01],
    [8.820e+02, 3.000e+00, 1.000e+00, 4.900e+01],
    [2.030e+03, 4.000e+00, 2.000e+00, 4.500e+01],
    [1.040e+03, 3.000e+00, 1.000e+00, 6.200e+01],
    [1.616e+03, 3.000e+00, 1.000e+00, 1.600e+01],
    [8.030e+02, 2.000e+00, 1.000e+00, 8.000e+01],
    [1.430e+03, 3.000e+00, 2.000e+00, 2.100e+01],
    [1.656e+03, 3.000e+00, 1.000e+00, 6.100e+01],
    [1.541e+03, 3.000e+00, 1.000e+00, 1.600e+01],
    [9.480e+02, 3.000e+00, 1.000e+00, 5.300e+01],
    [1.224e+03, 2.000e+00, 2.000e+00, 1.200e+01],
    [1.432e+03, 2.000e+00, 1.000e+00, 4.300e+01],
    [1.660e+03, 3.000e+00, 2.000e+00, 1.900e+01],
    [1.212e+03, 3.000e+00, 1.000e+00, 2.000e+01],
    [1.050e+03, 2.000e+00, 1.000e+00, 6.500e+01]])

y_train = np.array([300.   , 509.8  , 394.   , 540.   , 415.   , 230.   , 560.   ,
       294.   , 718.2  , 200.   , 302.   , 468.   , 374.2  , 388.   ,
       282.   , 311.8  , 401.   , 449.8  , 301.   , 502.   , 340.   ,
       400.282, 572.   , 264.   , 304.   , 298.   , 219.8  , 490.7  ,
       216.96 , 368.2  , 280.   , 526.87 , 237.   , 562.426, 369.8  ,
       460.   , 374.   , 390.   , 158.   , 426.   , 390.   , 277.774,
       216.96 , 425.8  , 504.   , 329.   , 464.   , 220.   , 358.   ,
       478.   , 334.   , 426.98 , 290.   , 463.   , 390.8  , 354.   ,
       350.   , 460.   , 237.   , 288.304, 282.   , 249.   , 304.   ,
       332.   , 351.8  , 310.   , 216.96 , 666.336, 330.   , 480.   ,
       330.3  , 348.   , 304.   , 384.   , 316.   , 430.4  , 450.   ,
       284.   , 275.   , 414.   , 258.   , 378.   , 350.   , 412.   ,
       373.   , 225.   , 390.   , 267.4  , 464.   , 174.   , 340.   ,
       430.   , 440.   , 216.   , 329.   , 388.   , 390.   , 356.   ,
       257.8  ])

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)

print(f"X_means: {X_means}\nX_stds: {X_stds}")

X_scaled = (X_train - X_means) / X_stds

b_init = 0
w_init = np.array([0, 0, 0, 0])

iterations = 1000
alpha = 1.0e-1

(w_final, b_final, J_hist) = gradient_descent(X_scaled, y_train, w_init, b_init, alpha, iterations)

print(f"w_final: {w_final}     b_final: {b_final}")

cost_x = np.linspace(0, iterations - 1, iterations)
fig, ax = plt.subplots()
ax.scatter(cost_x, J_hist)
plt.show()
