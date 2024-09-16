import matplotlib.pyplot as plt
import numpy as np

# This line is optional, only use if you're running on a remote server without a display
# plt.switch_backend('agg')


# Define the sigmoid function
def exp_sdf(sdf, center=0.0):
    return np.exp(sdf - center + 1e-7) * 0.01


def trip_sdf(sdf, center=0.0, s=0.6):
    return ((sdf - center) / s + 1e-7) ** 3


# Define the domain of x
x = np.linspace(0, 0.1, 1000)  # 1000 points between -1 and 1

# List of s values
thickness = np.array([0.04, 0.06, 0.07, 0.08]).astype(np.float32)
s_values = thickness * np.exp(-thickness / 6.0) * (100 ** (1 / 3.0))

for s in s_values:
    y = trip_sdf(x, s=s)
    plt.plot(x, y, label=f"s = {s}")

y = exp_sdf(x, center=0.03)
plt.plot(x, y, label=f"exp")

plt.xlabel("d")
plt.ylabel("f(d)")
plt.title("Plot of f(d) = sigmoid(d/s) for different values of s")
plt.legend()
plt.grid(True)

# To save the plot uncomment the next line
plt.savefig("checkouts/thickness_plot.png")

# To display the plot
# plt.show()
