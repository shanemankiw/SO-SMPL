import matplotlib.pyplot as plt
import numpy as np

# This line is optional, only use if you're running on a remote server without a display
# plt.switch_backend('agg')


# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Define the domain of x
x = np.linspace(-0.1, 0.1, 1000)  # 1000 points between -1 and 1

# List of s values
s_values = [1e-3, 0.01, 0.05]

for s in s_values:
    y = sigmoid(x / s)
    plt.plot(x, y, label=f"s = {s}")

plt.xlabel("d")
plt.ylabel("f(d)")
plt.title("Plot of f(d) = sigmoid(d/s) for different values of s")
plt.legend()
plt.grid(True)

# To save the plot uncomment the next line
plt.savefig("checkouts/sigmoid_plot.png")

# To display the plot
# plt.show()
