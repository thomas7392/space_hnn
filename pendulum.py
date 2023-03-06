from system import System
import numpy as np

class Pendulum_2D(System):
    '''
    A 2D mathematical pendulum problem
    '''
    def __init__(self, m, l, ID=None):

        # List of possisble coordinates with corresponding EoM
        self.coordinates_implemented = {"generalized": getattr(self, "eom_generalized"),
                                        "canonical": getattr(self, "Hamilton")}

        # Pendulum specific constants
        self.m = m
        self.l = l
        self.args = [self.m, self.l]

        # Inherit parent class
        super().__init__(ID=ID)

    #=============================
    # Canonical
    #=============================
    def Hamiltonian(self, state):
        q, p = state[..., 0], state[..., 1]
        T = (p**2)/(2*self.m*self.l**2)
        V = self.m * self.g* self.l*(1 - np.cos(q))
        return T + V

    def Hamilton(self, time, state):
        q, p = state[0], state[1]
        dqdt = p/(self.m*self.l**2)
        dpdt = -self.m * self.g * self.l * np.sin(q)

        return np.array([dqdt, dpdt])

    def dV(self, q):
        return -self.m * self.g * self.l * np.sin(q)

    def dT(self, p):
        return p/(self.m * self.l**2)

    #=============================
    # Generalized
    #=============================
    def eom_generalized(self, time, state):

        # Variable showing that this method is an EoM
        d_angle = state[1]
        dd_angle= -self.g * np.sin(state[0]) / self.l

        return np.array([d_angle, dd_angle])

    #===========================
    # Coordinate transformations
    #===========================
    def generalized_to_canonical(self, time, state):
        '''
        Convert the generlised coordinates to the canonical coordinates
        '''

        can = np.zeros(state.shape)
        can[..., 0] = state[...,0]
        can[..., 1] = state[...,1] * (self.m*self.l**2)

        return can

    def canonical_to_generalized(self, time, can):
        '''
        Convert the canonical coordinates to the generalized coordinates
        '''

        gen = np.zeros(can.shape)
        gen[..., 0] = can[..., 0]
        gen[..., 1] = can[..., 1] / (self.m * self.l**2)

        return gen
