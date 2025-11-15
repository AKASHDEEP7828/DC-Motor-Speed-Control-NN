#motor_model.py

import numpy as np

class DCMotor:
    def __init__(self):
        self.Ra = 1.2      # Armature resistance
        self.La = 0.5      # Armature inductance
        self.Ke = 0.45     # Back EMF constant
        self.Kt = 0.45     # Torque constant
        self.J = 0.01      # Moment of inertia
        self.B = 0.001     # Friction coefficient
        self.TL = 0        # Load torque

        self.ia = 0
        self.omega = 0

    def step(self, Va, dt):
        dia = (Va - self.Ra*self.ia - self.Ke*self.omega) / self.La
        self.ia += dia * dt

        domega = (self.Kt*self.ia - self.B*self.omega - self.TL) / self.J
        self.omega += domega * dt

        return self.omega
