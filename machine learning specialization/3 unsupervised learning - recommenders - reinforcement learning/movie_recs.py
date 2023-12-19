import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from joblib import dump


def cofiCostFunc(X, W, b, Ynorm, R, num_users, num_movies, lambda_):
    Y_hat = (tf.linalg.matmul(X, W) + b) * R.astype(float)
    dist = (Y_hat - Ynorm)**2
    regW = lambda_ / 2.0 * tf.reduce_sum(W**2)
    regX = lambda_ / 2.0 * tf.reduce_sum(X**2)
    return(0.5 * tf.reduce_sum(dist) + regW + regX)


df_ratings_raw = pd.read_csv("ml-latest-small/ratings_with_deryk.csv")
print(df_ratings_raw.tail())

userIds = df_ratings_raw.groupby('userId')['movieId'].count().index.values
movieIds = df_ratings_raw.groupby('movieId')['userId'].count().index.values

Y = np.zeros((len(movieIds), len(userIds)))

for row in df_ratings_raw.itertuples(index=True):
    Y[np.where(movieIds == row.movieId)[0], np.where(userIds == row.userId)[0]] = row.rating

R = Y > 0

numRatings = R.astype(int).sum(axis=1)

Y_means = Y.sum(axis=1) / numRatings
Ynorm = (Y - Y_means.reshape((-1, 1))) * R.astype(float)
(num_movies, num_users) = Y.shape
lambda_ = 1
num_features = 100
rng = np.random.default_rng()
X = tf.Variable(rng.standard_normal((num_movies, num_features)), dtype=tf.float64, name='X')
W = tf.Variable(rng.standard_normal((num_features, num_users)), dtype=tf.float64, name='W')
b = tf.Variable(rng.standard_normal((1, num_users)), dtype=tf.float64, name='b')

optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
for i in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofiCostFunc(X, W, b, Ynorm, R, num_users, num_movies, lambda_)
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    # Log periodically.
    if i % 20 == 0:
        print(f"Training loss at iteration {i}: {cost_value:0.1f}")

dump(X, "movie_rec_models/X.joblib")
dump(W, "movie_rec_models/W.joblib")
dump(b, "movie_rec_models/b.joblib")
dump(R, "movie_rec_models/R.joblib")
dump(Y_means, "movie_rec_models/Y_means.joblib")
dump(userIds, "movie_rec_models/userIds.joblib")
dump(movieIds, "movie_rec_models/movieIds.joblib")
