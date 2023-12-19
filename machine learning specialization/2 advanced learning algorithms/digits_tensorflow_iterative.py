import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential as TFSequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy


np.set_printoptions(precision=12)

X = np.load("./X.npy")
Y = np.load("./Y.npy")

data_blob = np.hstack([X, Y])

rng = np.random.default_rng(2)

rng.shuffle(data_blob, axis=0)

X = data_blob[:, :-1]
Y = data_blob[:, -1].reshape(-1, 1)

train_cv_split = 0.8
(m, n) = X.shape
train_set_size = int(m * train_cv_split)

X_train = X[0:train_set_size]
Y_train = Y[0:train_set_size]

X_cv = X[train_set_size:]
Y_cv = Y[train_set_size:]

(cv_set_size, _) = X_cv.shape

epochs = 100
alpha = 1e-3
lambda_ = 0.0001

error_cv_hist = []
error_train_hist = []
lambda_hist = []

for i in range(13):
    model = TFSequential(
        [
            Dense(units=120, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(lambda_)),
            Dense(units=40, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(lambda_)),
            Dense(units=10, activation="linear")
        ],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
        loss=SparseCategoricalCrossentropy(from_logits=True)
    )

    model.fit(X, Y, epochs=epochs)
    Y_cv_hat = np.argmax(model.predict(X_cv), axis=1).reshape(-1, 1)
    Y_train_hat = np.argmax(model.predict(X_train), axis=1).reshape(-1, 1)
    error_rate_cv = np.sum((Y_cv_hat != Y_cv.astype(int)).astype(int)) / cv_set_size * 100.
    error_rate_train = np.sum((Y_train_hat != Y_train.astype(int)).astype(int)) / train_set_size * 100.
    error_cv_hist.append(error_rate_cv)
    error_train_hist.append(error_rate_train)
    lambda_hist.append(lambda_)
    lambda_ *= 2

print(f"labmda_hist:\n{lambda_hist}")
print(f"error_train_hist:\n{error_train_hist}")
print(f"error_cv_hist:\n{error_cv_hist}")
