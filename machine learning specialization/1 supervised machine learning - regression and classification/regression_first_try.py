import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(a, w, b, sqft, price):
    partial_ans = w * sqft + b - price
    dJ_dw = 1.0 / sqft.shape[0] * np.sum(partial_ans * sqft)
    dJ_db = 1.0 / sqft.shape[0] * np.sum(partial_ans)

    w = w - a * dJ_dw
    b = b - a * dJ_db

    # Need to learn more about the learning rate.
    # Can it be dynamic? the b term needs a much higher value than w for this data setS
    # in order for the model to converge on the data correctly.

    return (w, b)


def calc_cost(w, b, sqft, price):
    cost = 1.0 / (2.0 * sqft.shape[0]) * np.sum((w * sqft + b - price) ** 2)
    return cost

sqft = np.array([1431.0, 1384.0, 1531.0, 1946.0, 2520.0, 1926.0, 3161.0, 1594.0, 3403.0, 2737.0, 2392.0, 1169.0, 2178.0, 1546.0, 1929.0, 1285.0])
price = np.array([515075.0, 499800.0, 547575.0, 682450.0, 869000.0, 675950.0, 1077325.0, 568050.0, 1155975.0, 939525.0, 827400.0, 429925.0, 757850.0, 552450.0, 676925.0, 467625.0])

sqft_mean = sqft.mean()
sqft_min = sqft.min()
sqft_max = sqft.max()
sqft = (sqft - sqft.mean()) / (sqft.max() - sqft.min())
# price = (price - price.mean()) / (price.max() - price.min())

w = 0
b = 0
alpha = 0.6

done = False
prev_cost = -1

count = 0

while not done:
    cost = calc_cost(w, b, sqft, price)
    print(f"Iteration {count}, w = {w}   b = {b}:\n  Cost = {cost}")
    count += 1
    if prev_cost < 0:
        prev_cost = cost
        (w, b) = gradient_descent(alpha, w, b, sqft, price)
        cost_history = np.array([cost])
        continue
    cost_diff = abs(cost - prev_cost) / prev_cost
    prev_cost = cost
    cost_history = np.append(cost_history, [cost])
    print(f"  Cost diff: {cost_diff}")
    if cost_diff < 0.000000001 or count > 10000:
        print("AAAAAAAAAA")
        done = True
    else:
        (w, b) = gradient_descent(alpha, w, b, sqft, price)

print(f"w: {w}    b: {b}")

min_sqft = np.amin(sqft)
max_sqft = np.amax(sqft)

model_sqft = np.linspace(min_sqft, max_sqft, 2)
model_cost = w * model_sqft + b

fig, ax = plt.subplots()
ax.scatter(sqft, price)
ax.plot(model_sqft, model_cost)

cost_x = np.linspace(0, count - 1, count)

print(f"x: {cost_x.shape}")
print(f"y: {cost_history.shape}")


fig2, ax2 = plt.subplots()
ax2.scatter(cost_x, cost_history)
plt.show()
