import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

X = np.load("./X.npy", allow_pickle=False)
Y = np.load("./y.npy", allow_pickle=False)
alpha = 1e-3
epochs = 100

model = Sequential(
    [
        Dense(name="L1", units=25, activation="relu"),
        Dense(name="L2", units=15, activation="relu"),
        Dense(name="L3", units=10, activation="linear")
    ]
)

model.compile(
    optimizer=Adam(learning_rate=alpha),
    loss=SparseCategoricalCrossentropy(from_logits=True)
)

(m, n) = X.shape
(_, u) = Y.shape

data_blob = np.hstack((X, Y))

rng = np.random.default_rng()
rng.shuffle(data_blob, axis=0)

X = data_blob[:, 0:n]
Y = data_blob[:, n:]

(m, n) = X.shape
(_, u) = Y.shape

train_ratio = 0.98
train_size = int(train_ratio * m)

X_train = X[0:train_size]
Y_train = Y[0:train_size]

X_test = X[train_size:]
Y_test = Y[train_size:]

model.fit(
    x=X_train,
    y=Y_train,
    epochs=epochs,
    use_multiprocessing=True,
    verbose=1,
    validation_data=(X_test, Y_test)
)

Y_test_hat = model.predict(X_test)
Y_test_hat = np.argmax(Y_test_hat, axis=1).astype(int)

print(f"Y_test_hat.shape: {Y_test_hat.shape}")
print(f"Y_test.shape: {Y_test.shape}")

print(f"Y_test[:, 0]:\n{Y_test[:, 0].astype(int)}")
print(f"Y_test_hat:\n{Y_test_hat}")

errors = np.not_equal(Y_test_hat, Y_test[:, 0]).astype(int)
print(f"errors:\n{errors}")
print(f"number of errors:\n{np.sum(errors)}")
print(f"error rate: {np.sum(errors) / len(errors)}")
