import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.expand_frame_repr', False)

X = load('movie_rec_models/X.joblib')
W = load('movie_rec_models/W.joblib')
b = load('movie_rec_models/b.joblib')
R = load('movie_rec_models/R.joblib')
Y_means = load('movie_rec_models/Y_means.joblib')
userIds = load('movie_rec_models/userIds.joblib')
movieIds = load('movie_rec_models/movieIds.joblib')

df_movie_names = pd.read_csv("ml-latest-small/movies.csv", index_col='movieId')

numRatings = R.astype(int).sum(axis=1)

# Get ratings
Y_hat = (tf.linalg.matmul(X, W) + b) * np.logical_not(R).astype(float) + Y_means.reshape(-1, 1)

deryk_index = np.argwhere(userIds == 611)[0][0]
df_deryk_Y_hat = pd.DataFrame(data={'prediction': Y_hat[:, deryk_index], 'public_rating': Y_means, 'num_ratings': numRatings}, index=movieIds)


# df_deryk_recs_raw = df_deryk_Y_hat[df_deryk_Y_hat[df_deryk_Y_hat['prediction'].gt(4)]['num_ratings'].gt(8)].sort_values('avg_rating', ascending=False)
df_deryk_recs_raw = df_deryk_Y_hat[((df_deryk_Y_hat['prediction'] >= 4) & (df_deryk_Y_hat['num_ratings'] >= 20))].sort_values('public_rating', ascending=False)
df_deryk_recs_raw = df_deryk_recs_raw.join(df_movie_names)
print(df_deryk_recs_raw[['prediction', 'public_rating', 'title']].head(20))

deryk_recs_movieIds = df_deryk_recs_raw.index.values
