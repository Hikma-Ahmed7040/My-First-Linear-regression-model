import numpy as np
import matplotlib.pyplot as plt

w = 1   # your starting guess
x = 2   # the input
y_true = 8     # the correct answer
alpha = 0.01             # learning rate

# To store Loss history for visualization
loss_history = []
for i in range(50): # you can change 50 to any number of iterations
      y_pred = w * x         # predicted output
      loss = (y_pred - y_true)**2  # loss function (MSE)
      dL_dw = 8 * w - 32  # derivative of the loss function w.r.t. w
      w = w - alpha * dL_dw  # update the weight using gradient descent
      loss_history.append(loss)  # store the loss for visualization
      print(f"step {i+1}: w = {w:.4f}, loss = {loss:.4f}")


# Plot loss over time
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss decreasing over time')
plt.grid(True)
plt.show()
