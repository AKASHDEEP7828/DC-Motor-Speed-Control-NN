import torch
import numpy as np
import matplotlib.pyplot as plt
from motor_model import DCMotor
from train_controller import NNController

model = NNController()
model.load_state_dict(torch.load("nn_controller.pth"))
model.eval()

motor = DCMotor()
desired_speed = 150
dt = 0.001
steps = 4000

speed_log = []

for t in range(steps):
    inp = torch.tensor([[desired_speed, motor.omega]], dtype=torch.float32)
    Va = model(inp).item()
    omega = motor.omega
    motor.step(Va, dt)
    speed_log.append(omega)

plt.plot(speed_log)
plt.xlabel("Time Steps")
plt.ylabel("Speed (rad/s)")
plt.title("NN Based DC Drive Speed Response")
plt.grid()
plt.savefig("results/speed_response.png")
plt.show()
