import gymnasium as gym
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE
from collections import deque, namedtuple


def sample_memory(memory_buffer, mini_batch_size):
    mini_batch = random.sample(memory_buffer, k=mini_batch_size)

    states = tf.convert_to_tensor(
        np.array([x.state for x in mini_batch if x is not None]),
        dtype=tf.float32
    )

    actions = tf.convert_to_tensor(
        np.array([x.action for x in mini_batch if x is not None]),
        dtype=tf.float32
    )

    rewards = tf.convert_to_tensor(
        np.array([x.reward for x in mini_batch if x is not None]),
        dtype=tf.float32
    )

    next_states = tf.convert_to_tensor(
        np.array([x.next_state for x in mini_batch if x is not None]),
        dtype=tf.float32
    )

    done_vals = tf.convert_to_tensor(
        np.array([x.done for x in mini_batch if x is not None]),
        dtype=tf.float32
    )

    return (states, actions, rewards, next_states, done_vals)


@tf.function
def agent_learn(experiences, q_net, q_hat_net, optimizer, tau):
    with tf.GradientTape() as tape:
        (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_done_vals
        ) = experiences

        proj_rewards = tf.reduce_max(q_hat_net(batch_next_states), axis=-1)
        y_q_hat = batch_rewards + (1 - batch_done_vals) * (gamma * proj_rewards)
        y_q = q_net(batch_states)
        # Get the y_q values based on the action *actually taken* in each sample
        # Slice y_q based on indices from batch_actions
        y_q = tf.gather_nd(y_q, tf.stack([tf.range(y_q.shape[0]), tf.cast(batch_actions, tf.int32)], axis=1))
        loss = MSE(y_q_hat, y_q)
        gradients = tape.gradient(loss, q_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_net.trainable_variables))
        for q_hat_net_weights, q_net_weights in zip(
            q_hat_net.weights, q_net.weights
        ):
            q_hat_net_weights.assign(tau * q_net_weights + (1.0 - tau) * q_hat_net_weights)


# Hyperparameters
memory_size = 100000        # Number of (state, artion, reward, next_state, done) tuples to store
alpha = 1e-3                # Learning rate
gamma = 0.985               # Discount factor
num_steps_for_update = 4    # Perform a learning update every C time steps
num_episodes = 100000         # Total training episodes to run
max_time_steps = 2000       # Maximum allowable time steps in a single episode
epsilon = 1.0               # Epsilon-greedy factor
epsilon_min = 0.01
epsilon_decay = 0.997
mini_batch_size = 128        # Size of training mini-batch
tau = 1e-3                  # Soft-update rate

q_net = Sequential([
    Input(shape=(8,)),
    Dense(units=96, activation='relu'),
    Dense(units=96, activation='relu'),
    Dense(units=4, activation='linear')
])

q_hat_net = Sequential([
    Input(shape=(8,)),
    Dense(units=96, activation='relu'),
    Dense(units=96, activation='relu'),
    Dense(units=4, activation='linear')
])

optimizer = Adam(learning_rate=alpha)
q_hat_net.set_weights(q_net.get_weights())

memory_buffer = deque(maxlen=memory_size)

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

env = gym.make("LunarLander-v2", enable_wind=True, wind_power=5.0, turbulence_power=0.2)
episode_rewards = deque(maxlen=100)
rando_portion = 0
time_steps_taken = 0

for i in range(num_episodes):
    state, info = env.reset()
    episode_reward = 0
    engines_disabled = False
    for t in range(max_time_steps):
        actions = q_net(np.expand_dims(state, axis=0))
        if state[6] >= 1.0 and state[7] >= 1.0:
            engines_disabled = True

        if engines_disabled:
            action = 0
        elif(random.random() <= epsilon):
            action = random.choice([0, 1, 2, 3])
            rando_portion += 1
        else:
            action = np.argmax(actions)
        time_steps_taken += 1
        # print(f"action: {action}")
        next_state, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        memory_buffer.append(experience(state, action, reward, next_state, (terminated or truncated)))
        if (t + 1) % num_steps_for_update == 0 and len(memory_buffer) > mini_batch_size:
            experiences = sample_memory(memory_buffer, mini_batch_size)
            agent_learn(experiences, q_net, q_hat_net, optimizer, tau)

        if terminated or truncated:
            break
        state = next_state.copy()
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(episode_reward)
    time_steps_taken = 0
    rando_portion = 0
    prev_100_avg = sum(episode_rewards) / len(episode_rewards)
    print(f"episode {i}: reward={episode_reward}, prev_100_avg: {prev_100_avg}, epsilon: {epsilon}")
    if prev_100_avg > 250:
        env = gym.make("LunarLander-v2", render_mode="human", enable_wind=True, wind_power=5.0, turbulence_power=0.2)
        epsilon = 0
        epsilon_min = 0

env.close()
