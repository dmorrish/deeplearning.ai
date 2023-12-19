from joblib import load
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

model = load("movie_recs_content_models/model.joblib")
scalerUsers = load("movie_recs_content_models/scalerUsers.joblib")
scalerMovies = load("movie_recs_content_models/scalerMovies.joblib")
scalerY = load("movie_recs_content_models/scalerY.joblib")
df_movies = load("movie_recs_content_models/df_movies.joblib")
df_movies_raw = load("movie_recs_content_models/df_movies_raw.joblib")
df_ratings_raw = load("movie_recs_content_models/df_ratings_raw.joblib")
df_users_raw = load("movie_recs_content_models/df_users_raw.joblib")

uid = 611
rated_locs = df_ratings_raw['userId'] == uid
movies_rated = df_ratings_raw.loc[rated_locs]['movieId']

movie_vecs = df_movies.drop(movies_rated, axis=0).reset_index()
user_row = pd.DataFrame(df_users_raw.loc[uid]).transpose()
user_vecs = pd.DataFrame(np.repeat(user_row.values, movie_vecs.values.shape[0], axis=0))
user_vecs.columns = user_row.columns
user_vecs['userId'] = uid
cols = user_vecs.columns.to_list()
cols = cols[-1:] + cols[:-1]
user_vecs = user_vecs[cols]

scaled_movie_vecs = scalerMovies.transform(movie_vecs)
scaled_user_vecs = scalerUsers.transform(user_vecs)

y_p_scaled = model.predict([scaled_user_vecs[:, 3:], scaled_movie_vecs[:, 2:]])
y_p = scalerY.inverse_transform(y_p_scaled)

movie_titles = df_movies_raw.reset_index()[['movieId', 'title']]

movie_recs = movie_vecs
movie_recs['pred_rating'] = y_p

movie_recs = pd.merge(movie_recs, movie_titles, on='movieId')
movie_recs = movie_recs.sort_values("pred_rating", ascending=False)
movie_recs = movie_recs.loc[movie_recs['rating_count'] > 20]
print(movie_recs.head(10))
