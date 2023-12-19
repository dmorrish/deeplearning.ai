import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from joblib import dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import random


def strip_year(title):
    begin = title.rfind('(')
    end = title.rfind(')')
    year = 0

    if(begin > 0 and end > 0):
        year_str = title[begin + 1: end]
        if year_str.isnumeric():
            year = int(year_str)
    return year


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df_ratings_raw = pd.read_csv("ml-latest-small/ratings_with_deryk.csv")
df_movies_raw = pd.read_csv("ml-latest-small/movies.csv", index_col="movieId")

############################
# Prep movie training data #
############################
df_agg_rating_count = df_ratings_raw.groupby("movieId").count()['rating'].astype('Int64')
df_agg_rating_count = df_agg_rating_count.rename("rating_count")
df_agg_rating_avg = df_ratings_raw.groupby("movieId").mean()['rating']
df_agg_rating_avg = df_agg_rating_avg.rename("rating_avg")

df_movies_raw = df_movies_raw.join(df_agg_rating_count)
df_movies_raw = df_movies_raw.join(df_agg_rating_avg)

df_movies_raw['rating_count'] = df_movies_raw['rating_count'].fillna(0)
df_movies_raw['rating_avg'] = df_movies_raw['rating_avg'].fillna(2.5)

df_movies_raw['year'] = df_movies_raw['title'].map(strip_year)

genre_list = []

for row_tuple in df_movies_raw.itertuples(index=True):
    genres = row_tuple.genres.split("|")
    for genre in genres:
        if genre not in genre_list and genre != '(no genres listed)':
            genre_list.append(genre)

genre_list.sort()

for genre in genre_list:
    df_movies_raw[genre] = df_movies_raw['genres'].str.split(pat="|").apply(lambda x: genre in x)

df_movies = df_movies_raw.drop('genres', axis=1)
df_movies = df_movies.drop('title', axis=1)
# print(df_movies.head())

###########################
# Prep user training data #
###########################
df_agg_rating_count = df_ratings_raw.groupby("userId").count()['rating'].astype('Int64')
df_agg_rating_count = df_agg_rating_count.rename("rating_count")
df_agg_rating_avg = df_ratings_raw.groupby("userId").mean()['rating']
df_agg_rating_avg = df_agg_rating_avg.rename("rating_avg")

df_users_raw = pd.DataFrame(df_agg_rating_count)
df_users_raw = df_users_raw.join(df_agg_rating_avg)

df_movies_full = df_ratings_raw.join(df_movies, on='movieId')

for genre in genre_list:
    df_genre_ratings = df_movies_full[['rating', 'userId']].loc[df_movies_full[genre] == True].groupby(['userId']).mean()
    df_genre_ratings = df_genre_ratings.rename(columns={"rating": genre})
    df_users_raw = df_users_raw.join(df_genre_ratings)

df_users_raw = df_users_raw.fillna(0)
# print(df_users_raw.head())

y = df_movies_full['rating'].to_numpy()
df_movies_full = df_movies_full.drop(['userId', 'rating', 'timestamp'], axis=1)

df_users_full = df_ratings_raw.join(df_users_raw, on='userId')
df_users_full = df_users_full.drop(['movieId', 'rating', 'timestamp'], axis=1)
# print("############\n############\n############")
# print(df_movies_full.head(20))
# print(df_users_full.head(20))
# print(y.head(20))

scalerMovies = StandardScaler()
scalerMovies.fit(df_movies_full)
movies_full = scalerMovies.transform(df_movies_full)

scalerUsers = StandardScaler()
scalerUsers.fit(df_users_full)
users_full = scalerUsers.transform(df_users_full)

scalerY = MinMaxScaler((-1, 1))
scalerY.fit(y.reshape((-1, 1)))
y = scalerY.transform(y.reshape((-1, 1)))

# print("############\n############\n############")
# print(movies_full[0:20])
# print(users_full[0:20])
# print(y[0:20])

random_state = random.randint(0, 2000000)

movies_train, movies_cv = train_test_split(movies_full, train_size=0.8, shuffle=True, random_state=random_state)
users_train, users_cv = train_test_split(users_full, train_size=0.8, shuffle=True, random_state=random_state)
y_train, y_cv = train_test_split(y, train_size=0.8, shuffle=True, random_state=random_state)

num_user_features = users_train.shape[1] - 3
num_movie_features = movies_train.shape[1] - 2

num_outputs = 32
tf.random.set_seed(random_state)

user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=num_outputs, activation='linear')
])

movie_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=num_outputs, activation='linear')
])

input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

input_movie = tf.keras.layers.Input(shape=(num_movie_features))
vm = movie_NN(input_movie)
vm = tf.linalg.l2_normalize(vm, axis=1)

output = tf.keras.layers.Dot(axes=1)([vu, vm])
model = tf.keras.Model([input_user, input_movie], output)

# model.summary()

tf.random.set_seed(random_state)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)

tf.random.set_seed(random_state)
model.fit([users_train[:, 3:], movies_train[:, 2:]], y_train, epochs=30)

model.evaluate([users_cv[:, 3:], movies_cv[:, 2:]], y_cv)

dump(model, "movie_recs_content_models/model.joblib")
dump(scalerUsers, "movie_recs_content_models/scalerUsers.joblib")
dump(scalerMovies, "movie_recs_content_models/scalerMovies.joblib")
dump(scalerY, "movie_recs_content_models/scalerY.joblib")
dump(df_movies, "movie_recs_content_models/df_movies.joblib")
dump(df_movies_raw, "movie_recs_content_models/df_movies_raw.joblib")
dump(df_ratings_raw, "movie_recs_content_models/df_ratings_raw.joblib")
dump(df_users_raw, "movie_recs_content_models/df_users_raw.joblib")
