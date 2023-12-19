import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import json
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
data = pd.read_csv("heart.csv")

columns_to_encode = [
    'Sex',
    'ChestPainType',
    'RestingECG',
    'ExerciseAngina',
    'ST_Slope'
]

data_encoded = pd.get_dummies(data, prefix=columns_to_encode, columns=columns_to_encode)

features = [x for x in data_encoded.columns if x != 'HeartDisease']

X_train, X_cv, Y_train, Y_cv = train_test_split(data_encoded[features], data_encoded['HeartDisease'], test_size=0.2)

min_samples_split_list = [35, 35, 35, 35, 35, 35, 35]
max_depth_list = [7, 7, 7, 7, 7, 7, 7, 7]
n_estimators_list = [100, 100, 100]

min_samples_split_list = [35]
max_depth_list = [7]
n_estimators_list = [100]

decision_tree_training_results = []
decision_tree_model_count = 0

decision_tree_training_accuracies = []
decision_tree_cv_accuracies = []
decision_tree_indices = []

for min_samples_split in min_samples_split_list:
    for max_depth in max_depth_list:
        for n_estimators in n_estimators_list:
            model = RandomForestClassifier(
                min_samples_split=min_samples_split,
                max_depth=max_depth,
                n_estimators=n_estimators,
                n_jobs=8
            )
            model.fit(X_train, Y_train)
            Y_hat_cv = model.predict(X_cv)
            Y_hat_train = model.predict(X_train)

            print(f"random forest train accuracy: {accuracy_score(Y_train, Y_hat_train)}")
            print(f"random forest cv accuracy: {accuracy_score(Y_cv, Y_hat_cv)}")
            new_model_result = {
                "index": decision_tree_model_count,
                "min_samples_split": min_samples_split,
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "accuracy_train": accuracy_score(Y_train, Y_hat_train),
                "accuracy_cv": accuracy_score(Y_cv, Y_hat_cv)
            }
            decision_tree_training_results.append(new_model_result)
            decision_tree_training_accuracies.append(accuracy_score(Y_train, Y_hat_train))
            decision_tree_cv_accuracies.append(accuracy_score(Y_cv, Y_hat_cv))
            decision_tree_indices.append(decision_tree_model_count)
            decision_tree_model_count += 1

with open("decision_tree_models.txt", "w") as f_models:
    json.dump(decision_tree_training_results, f_models, indent=4)

epochs = 1000
lambdas = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
l2_unit_list = [6, 10, 14, 18, 22]
alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-3]

neural_network_training_results = []
neural_network_model_count = 0

neural_network_training_accuracies = []
neural_network_cv_accuracies = []
neural_network_indices = []

for lambda_ in lambdas:
    for l2_units in l2_unit_list:
        for alpha in alphas:
            model = Sequential(
                [
                    Dense(
                        units=20,
                        activation="relu",
                        name="L1",
                        kernel_regularizer=tf.keras.regularizers.L2(lambda_)
                    ),
                    Dense(
                        units=l2_units,
                        activation="relu",
                        name="L2",
                        kernel_regularizer=tf.keras.regularizers.L2(lambda_)
                    ),
                    Dense(
                        units=2,
                        activation="linear",
                        name="L3"
                    )
                ]
            )

            model.compile(
                optimizer=Adam(learning_rate=alpha),
                loss=SparseCategoricalCrossentropy(from_logits=True)
            )

            model.fit(X_train.to_numpy().astype(float), Y_train.to_numpy().astype(int), epochs=epochs, verbose=0)
            Y_hat_train = np.argmax(model.predict(X_train.astype(float)), axis=1).reshape(-1, 1)
            Y_hat_cv = np.argmax(model.predict(X_cv.astype(float)), axis=1).reshape(-1, 1)

            print(f"{neural_network_model_count + 1} of {len(lambdas) * len(l2_unit_list) * len(alphas)}")
            print(f"neural network train accuracy: {accuracy_score(Y_train, Y_hat_train)}")
            print(f"neural network cv accuracy: {accuracy_score(Y_cv, Y_hat_cv)}")
            new_model_result = {
                "index": neural_network_model_count,
                "lambda": lambda_,
                "l2_units": l2_units,
                "alpha": alpha,
                "accuracy_train": accuracy_score(Y_train, Y_hat_train),
                "accuracy_cv": accuracy_score(Y_cv, Y_hat_cv)
            }
            neural_network_training_results.append(new_model_result)
            neural_network_training_accuracies.append(accuracy_score(Y_train, Y_hat_train))
            neural_network_cv_accuracies.append(accuracy_score(Y_cv, Y_hat_cv))
            neural_network_indices.append(neural_network_model_count)
            neural_network_model_count += 1

with open("neural_network_models.txt", "w") as f_models:
    json.dump(neural_network_training_results, f_models, indent=4)

fig, ax = plt.subplots()
ax.plot(decision_tree_indices, decision_tree_training_accuracies, label="Training Set")
ax.plot(decision_tree_indices, decision_tree_cv_accuracies, label="Cross-Validation Set")
ax.legend()
fig.suptitle("Decision Tree Training Results")

fig2, ax2 = plt.subplots()
ax2.plot(neural_network_indices, neural_network_training_accuracies, label="Training Set")
ax2.plot(neural_network_indices, neural_network_cv_accuracies, label="Cross-Validation Set")
ax2.legend()
fig2.suptitle("Neural Network Training Results")
plt.show()
