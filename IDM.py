import numpy as np
import math
import os 
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class IDM():
    def __init__(self):
        self.a = 1.32 # maximum acceleration [m/s^2]
        self.b = 2.3 # braking deceleration
        self.T = 1.85 
        self.s0 = 5.2 # minimum distance [m]
        self.v0 = 50/3.6 # v_limit [m/s]
        self.delta = 4
        self.v_alpha = 0 # ego velocity

    def get_accel(self, distance, rel_vel):
        first_term = math.pow(self.v_alpha/self.v0, self.delta)
        s_star = self.s0 + self.v_alpha*self.T + self.v_alpha*rel_vel / (2 * math.sqrt(self.a*self.b))
        second_term = (s_star / distance) ** 2

        accel = self.a * (1 - first_term - second_term)
        
        return accel

    def update(self, ego_vy):
        self.v_alpha = ego_vy

'''
FROM WIKIPEDIA
:param v0:
    max speed of the car
:param delta:
    technical term in the acc calculation
:param a:
    max acceleration m/s^2
:param b:
    normal deceleration m/s^2
:param s0: default 2
    least safe distance between two cars
    --> exceed this means need to brake immediately
:param T:  default 1.5
    human reaction time T for s*
'''