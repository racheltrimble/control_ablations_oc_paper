from casadi import *


def get_cvodes(state, control_vector, dynamics, objective, timestep):
    # Formulate discrete time dynamics
    # CVODES from the SUNDIALS suite
    dae = {'x': state,
           'p': control_vector,
           'ode': dynamics,
           'quad': objective}
    return integrator('F', 'cvodes', dae, 0, timestep)


# Pulling out duplicate code from the functions below...
def functional_integrator_setup(state, control_vector, dynamics, objective, timestep, steps_per_interval):
    dt = timestep / steps_per_interval
    f = Function('f', [state, control_vector], [dynamics, objective])
    x0 = MX.sym('X0', state.shape)
    control = MX.sym('U', control_vector.shape)
    running_state = x0
    cost = 0
    return dt, f, x0, control, running_state, cost


# Fixed step Runge-Kutta 4 integrator
def get_rk4(state, control_vector, dynamics, objective, timestep, steps_per_interval):
    dt, f, x0, control, running_state, cost = functional_integrator_setup(state,
                                                                          control_vector,
                                                                          dynamics,
                                                                          objective,
                                                                          timestep,
                                                                          steps_per_interval)
    for j in range(steps_per_interval):
        k1, k1_q = f(running_state, control)
        k2, k2_q = f(running_state + dt / 2 * k1, control)
        k3, k3_q = f(running_state + dt / 2 * k2, control)
        k4, k4_q = f(running_state + dt * k3, control)
        running_state = running_state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        cost = cost + dt / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    return Function('F', [x0, control], [running_state, cost], ['x0', 'p'], ['xf', 'qf'])


# Based on description here:
# http://www.physics.drexel.edu/~steve/Courses/Comp_Phys/Integrators/simple.html
def get_midpoint(state, control_vector, dynamics, objective, timestep, steps_per_interval):
    dt, f, x0, control, running_state, cost = functional_integrator_setup(state,
                                                                          control_vector,
                                                                          dynamics,
                                                                          objective,
                                                                          timestep,
                                                                          steps_per_interval)
    for j in range(steps_per_interval):
        k1, k1_q = f(running_state, control)
        dx = dt * k1
        k_mid, k_mid_q = f(running_state + dx / 2, control)
        running_state = running_state + dt * k_mid
        cost = cost + dt * k_mid_q
    return Function('F', [x0, control], [running_state, cost], ['x0', 'p'], ['xf', 'qf'])
