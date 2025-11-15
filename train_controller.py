import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from motor_model import DCMotor

# Neural Network Controller
class NNController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# Generate STABLE training data
motor = DCMotor()
dt = 0.001

X = []
Y = []

# Synthetic stable controller (simple proportional control)
K = 0.8

for step in range(5000):
    desired_speed = 150
    current_speed = motor.omega
    error = desired_speed - current_speed

    # simple stable voltage
    Va = K * error

    # clamp
    Va = max(min(Va, 100), -100)

    # collect training data
    X.append([desired_speed, current_speed])
    Y.append([Va])

    motor.step(Va, dt)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

# Train NN
model = NNController()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item()}")

torch.save(model.state_dict(), "nn_controller.pth")
print("Training completed and model saved!")
