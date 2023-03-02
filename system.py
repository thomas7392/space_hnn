import numpy as np;
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

import math
import time

import matplotlib.pyplot as plt

class System:
    '''
    Class to be used as parent class of a more specific dynamical system
    This class contains integrators to solve the dynamical system that apply to
    all types of dynamical systems. Also contains plotting functionality
    '''

    def __init__(self, ID=None):

        self.ID = ID

        # Physical constants
        self.g = 9.81
        self.mu_sun = 3.986004418e14


    def set_initial_condition(self,
                              time,
                              state,
                              coordinates,
                              y0_all_coordinates = True):
        '''
        Set initial condition
        '''

        # Check that coordinate system is implemented in class
        assert coordinates in self.coordinates_implemented.keys(), f"No EoM for coordinates implemented,\
              implemented coordinates:\n {self.coordinates_implemented.keys()}"

        # Remove previous solution in instance
        self.clear_solution(verbose = False)

        # Include initial state
        y0 = np.append([time], state)
        self.y0 = {coordinates: y0}

        # Include initial state in terms of all implemented coordinate systems
        if y0_all_coordinates:
            self.store_initial_values(y0, coordinates)


    def store_initial_values(self, y0, y0_coordinates):
        '''
        Convert initial coordinates to all other coordinate systems
        '''
        for coordinates in self.coordinates_implemented.keys():
            if coordinates != y0_coordinates:
                transformation = getattr(self, f"{y0_coordinates}_to_{coordinates}")
                self.y0[coordinates] = np.append(y0[0], transformation(y0[0], y0[1:]))

    def clear_solution(self, verbose = True):
        '''
        Clear all atributes related to a solutino obtained via integration
        '''
        if hasattr(self, 'states'):
            del self.states
            del self.evals
            del self.integrator
            del self.time_step
            del self.propagator
            del self.rtol
            del self.atol
            del self.hamiltonian
            del self.time

            if verbose:
                print("clear_solution: solution is cleared")
        else:
            if verbose:
                print("clear_solution: no solution found")


    def calculate_coordinates(self, coordinates):

        assert hasattr(self, 'states'), "no solution found, cant plot position vs time"
        assert coordinates in self.coordinates_implemented.keys(), f"Coordinate not implemented,\
              implemented coordinates:\n {self.coordinates_implemented.keys()}"

        known_coordinates = list(self.states.keys())[0]
        transformation = getattr(self, f"{known_coordinates}_to_{coordinates}")

        self.states[coordinates] = np.append(self.states[known_coordinates][:,0].reshape(-1, 1),
                                             transformation(self.states[known_coordinates][:,0],
                                                  self.states[known_coordinates][:,1:]), axis=1)



    def calculate_hamiltonian(self):

        assert hasattr(self, "Hamiltonian"), "Hamiltonian method not implemented"
        assert hasattr(self, 'states'), "No solution found, cant calculate hamiltonian values"
        assert "canonical" in list(self.coordinates_implemented.keys()), "Canonical\
              coordinates not implemented"

        # If solutino is not calculated in terms of canonical coordiantes, calculate them
        if "canonical" not in list(self.states.keys()):
            self.calculate_coordinates("canonical")

        self.hamiltonian = self.Hamiltonian(self.states['canonical'][:,1:])


    def integrate(self,
                  method,
                  tf,
                  coordinates,
                  h=None,
                  rtol = None,
                  atol = None):
        '''
        Integrate an IVP
        '''

        # Check if an EoM is implemented for the chosen coordinates, if
        # initial state avilable in correct coordinates and whether solution
        # already exists.
        self.implemented_integrators = ["RK4", "euler", "leapfrog", "dop853 (scipy.integrate.solve_ivp)"]


        assert hasattr(self, "y0"), "Initial condition not set, use <instance_name>.set_initial_condition(time, state, coordinate_system)"
        assert coordinates in self.coordinates_implemented.keys(), f"No EoM for coordinates implemented,\
              implemented coordinates:\n {self.coordinates_implemented.keys()}"
        assert coordinates in self.y0.keys(), "Initial conditions not calculated for coordinates"
        assert not hasattr(self, 'states'), "This case is already integrated, a solution exists"

        self.states = {}
        self.integrator = method
        self.propagator = self.coordinates_implemented[coordinates]
        temp_y0 = self.y0[coordinates]

        start_time = time.time()
        if method == "RK4":

            # Method Specific Assertions
            assert h != None, "h argument mandetory: choose constant time step"

            # Prepare specific integrator
            fun = self.coordinates_implemented[coordinates]
            self.time_step = h

            # Integrate
            self.states[coordinates] = self.rk4(fun, temp_y0, tf, h)

        elif method == "euler":

            # Method Specific Assertions
            assert h != None, "h argument mandetory: choose constant time step"

            fun = self.coordinates_implemented[coordinates]
            self.time_step = h
            self.states[coordinates] = self.euler(fun, temp_y0, tf, h)

        elif method == "leapfrog":

            assert h != None, "h argument mandetory: choose constant time step"
            assert hasattr(self, "dV"), "dV method mandetory, leapfrog only works when hamiltonian is serpable"
            assert hasattr(self, "dT"), "dT method mandetory, leapfrog only works when hamiltonian is serpable"

            self.time_step = h
            self.states[coordinates] = self.leapfrog(self.dV, self.dT, temp_y0, tf, h)

        elif method == "dop853":
            assert rtol != None, "rtol argument mandetory: choose constant time step"
            assert atol != None, "atol argument mandetory: choose constant time step"

            fun = self.coordinates_implemented[coordinates]
            self.atol = atol
            self.rtol = rtol
            self.states[coordinates] = self.dop853(fun, temp_y0, tf, atol, rtol)
        else:

            raise ValueError(f"Integrator not implemented. Options: {self.implemented_integrators}")

        end_time = time.time()
        self.time = end_time - start_time

    def rk4(self, fun, y0, tf, h):
        '''
        Runge-Kutta Constant time step integrator
        '''

        # Prepare times
        t0 = y0[0]
        steps = math.ceil((tf - t0)/h)
        times = np.linspace(t0, tf, steps)

        # Prepare arr with states
        temp_states = np.zeros((steps, len(y0)))
        temp_states[0] = y0
        temp_states[:,0] = times

        # Analytically determine function evaluations
        self.evals = 4*(len(times) - 1)

        # Integrate
        for i, state in enumerate(temp_states[:-1]):

            k1 = fun(state[0], state[1:])
            k2 = fun(state[0] + h/2, state[1:] + h*k1/2)
            k3 = fun(state[0] + h/2, state[1:] + h*k2/2)
            k4 = fun(state[0] + h, state[1:] + h*k3)

            temp_states[i+1,1:] = state[1:] + (1./6.)*(k1 + 2*k2 + 2*k3 + k4)*h

        return temp_states

    def euler(self, fun, y0, tf, h):

        # Prepare times
        t0 = y0[0]
        steps = math.ceil((tf - t0)/h)
        times = np.linspace(t0, tf, steps)

        # Prepare arr with states
        temp_states = np.zeros((steps, len(y0)))
        temp_states[0] = y0
        temp_states[:,0] = times

        # Analytically determine function evaluations
        self.evals = len(times) - 1

        for i, state in enumerate(temp_states[:-1]):
            temp_states[i+1,1:] = state[1:] + h * fun(state[0],state[1:])

        return temp_states

    def leapfrog(self, dV, dT, y0, tf, h):

        # Prepare times
        t0 = y0[0]
        steps = math.ceil((tf - t0)/h)
        times = np.linspace(t0, tf, steps)

        # Prepare arr with states
        temp_states = np.zeros((steps, len(y0)))
        temp_states[0] = y0
        temp_states[:,0] = times

        d = len(y0)

        # Analytically determine function evaluations
        self.evals = 2*(len(times) - 1)

        for i, state in enumerate(temp_states[:-1]):

            p_old = state[int(1+d/2):]
            q_old = state[1:int(1+d/2)]

            p_temp = p_old + (1./2.) * h * dV(q_old)
            q_new  = q_old + h * dT(p_temp)

            temp_states[i+1, 1:int(1+d/2)] = q_new
            temp_states[i+1, 1+int(d/2):] = p_temp + (1./2.) * h * dV(q_new)

        return temp_states

    def dop853(self, fun, y0, tf, atol, rtol):

        result = solve_ivp(fun,
                           (y0[0], tf),
                           y0[1:],
                           atol=atol,
                           rtol=rtol,
                           method = "DOP853")

        time = result.t
        y = result.y
        self.evals = result.nfev

        # Put with time in array
        temp_states = np.zeros((len(y[0]), len(y)+1))
        temp_states[:,0] = time
        temp_states[:,1:] = y.T

        return temp_states

    def plot_variable(self, index, coordinates,
                      label = None,
                      linestyle = None,
                      ax = None):

        assert hasattr(self, 'states'), "no solution found, cant plot position vs time"

        if not ax:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.states[coordinates][:,0], self.states[coordinates][:,index],
                 label = label,
                 linestyle = linestyle)

        if label != None:
            ax.legend()

        return ax








