import numpy as np

from casadi import SX, MX
from control_ablations.generic_targets import OptimalControlCollocation


def generate_two_node_dynamics():
    # Declare state variables
    s0 = SX.sym('S0')
    s1 = SX.sym('S1')
    i0 = SX.sym('I0')
    i1 = SX.sym('I1')
    r0 = SX.sym('R0')
    r1 = SX.sym('R1')
    state_list = [s0, i0, r0, s1, i1, r1]

    # Declare control
    b0 = SX.sym('b0')
    control_list = [b0]

    # Model equations
    beta00 = 0.5
    beta01 = 0.1
    beta10 = 0.1
    beta11 = 0.5
    gamma = 0.2
    budget = 0.4

    s_to_i1 = s0 * i0 * beta00 + s0 * i1 * beta01
    s_to_i2 = s1 * i1 * beta11 + s1 * i0 * beta10
    i_to_r1 = gamma * i0 + b0 * i0 / (i0 + s0)
    i_to_r2 = gamma * i1 + (budget - b0) * i1 / (i1 + s1)

    xdot = [-s_to_i1,
            s_to_i1 - i_to_r1,
            i_to_r1,
            -s_to_i2,
            s_to_i2 - i_to_r2,
            i_to_r2]

    # Control bounds
    control_bounds_lower = [0]
    control_bounds_upper = [budget]

    # Objective term
    objective = i0 + i1 + r0 + r1

    s2_init = 1.0
    s1_init = 0.9
    initial_state = [s1_init, 1.0 - s1_init, 0.0, s2_init, 1.0 - s2_init, 0.0]
    state_bounds_lower = [0.0] * 6
    state_bounds_upper = [1.0] * 6

    options = {'state_list': state_list,
               "control_list": control_list,
               "dynamics_list": xdot,
               "objective": objective,
               "final_cost": None,
               "initial_state": initial_state,
               "control_bounds_lower": control_bounds_lower,
               "control_bounds_upper": control_bounds_upper,
               "state_bounds_lower": state_bounds_lower,
               "state_bounds_upper": state_bounds_upper,
               "inequality_constraint_function": None}
    return options


def test_two_nodes_optimal_control(graph=False):
    options = generate_two_node_dynamics()
    control_intervals = 40

    opt = OptimalControlCollocation(**options,
                                    time_horizon=5,
                                    control_intervals=control_intervals,
                                    integrator_type='cvodes')

    # Evaluate at a test point
    success, u_opt, x_opt = opt.run()

    if graph:
        state_line_styles = ['--', '.', '-'] * 2
        control_line_styles = ['-.']*2
        opt.plot(state_line_styles, control_line_styles)
    assert success
    assert (len(u_opt[0]) == control_intervals)


def test_robust_to_control_intervals(graph=False):
    options = generate_two_node_dynamics()

    opt = OptimalControlCollocation(**options,
                                    time_horizon=5,
                                    control_intervals=20,
                                    integrator_type='cvodes')

    # Evaluate at a test point
    success_20, u_opt_20, x_opt_20 = opt.run()

    if graph:
        state_line_styles = ['--', '.', '-'] * 2
        control_line_styles = ['-.']*2
        opt.plot(state_line_styles, control_line_styles)

    opt = OptimalControlCollocation(**options,
                                    time_horizon=5,
                                    control_intervals=40,
                                    integrator_type='cvodes')

    # Evaluate at a test point
    success_40, u_opt_40, x_opt_40 = opt.run()

    if graph:
        state_line_styles = ['--', '.', '-'] * 2
        control_line_styles = ['-.']*2
        opt.plot(state_line_styles, control_line_styles)

    assert success_20
    assert (len(u_opt_20[0]) == 20)
    assert success_40
    assert (len(u_opt_40[0]) == 40)
    for step in range(20):
        # Large tolerance as we expect some numerical error.
        assert (np.isclose(u_opt_20.transpose()[step], u_opt_40.transpose()[step*2], atol=0.1))


def test_collocation_constraints():
    options = generate_two_node_dynamics()

    opt = OptimalControlCollocation(**options,
                                    time_horizon=1,
                                    control_intervals=2,
                                    integrator_type='cvodes',
                                    polynomial_degree=1)
    opt.setup_collocation()
    opt.generate_nlp()
    # "What can we wobble about to optimise?" 1 control variable * 2 timesteps
    # 1 collocation point for 6 variables at 2 timesteps
    # 1 continuity point for 6 variables at 2 timesteps
    assert (len(opt.optimiser_input_variables_per_timestep) == 6)
    # 1 reward
    assert (opt.reward_accumulator.shape == (1, 1))
    # bounds as per inputs...
    assert (len(opt.optimiser_input_lower_bounds_per_timestep) == 2 + 12 + 12)
    # No budget constraint for the simple problem (0)
    # 2 collocations
    # 2 continuity. Last continuity is questionable but means the last state can't just go awol...
    assert (len(opt.g) == 4)


def generate_multi_node(num_nodes,
                        node_population=1.0,
                        cull_vs_thin=False,
                        host_scaling=1.0,
                        horizon=20,
                        steps=20):
    final_reward_only = True
    node_population = node_population * host_scaling
    if cull_vs_thin:
        # remote_beta = 0.01 / node_population
        remote_beta = 0.25 / node_population
        local_beta = 1.0 / node_population
        # budget_per_node = 0.125
        budget = 1.0
        # Uninfected nodes are bigger.
        initial_state_per_node = [node_population, 0.0]
        initial_state_per_node_seed = [0.2 * node_population, 0.8 * node_population]
        cost_per_action = 500 / 100 / node_population
    else:
        remote_beta = 0.1 / node_population
        local_beta = 0.5 / node_population
        budget_per_node = 1
        budget = budget_per_node * num_nodes * node_population
        initial_state_per_node = [node_population, 0.0]
        initial_state_per_node_seed = [0.9 * node_population, 0.1 * node_population]
        cost_per_action = 1000
    gamma = 0.2
    state_bounds_lower_per_node = [0.0, 0.0]
    # state_bounds_upper_per_node = [1.0 * node_population, 1.0 * node_population]
    state_bounds_upper_per_node = [np.inf, np.inf]
    # Declare model variables
    s_list = []
    i_list = []
    control_list = []
    control_line_type = []
    state_list = []
    state_line_type = []
    show_plots = []
    betas = np.ones((num_nodes, num_nodes))
    betas = betas * remote_beta
    initial_state = []

    for node in range(num_nodes):
        s = MX.sym('S' + str(node))
        i = MX.sym('I' + str(node))
        s_list.append(s)
        i_list.append(i)
        state_list += [s, i]
        state_line_type += ['--', '.']
        control_list.append(MX.sym('cull' + str(node)))
        control_line_type += ['-.']
        if cull_vs_thin:
            control_list.append(MX.sym('thin' + str(node)))
            control_line_type += ['-.']
        if node == 0 or node == 1:
            show_plots += [True, True]
        else:
            show_plots += [False, False]
        betas[node, node] = local_beta
        if node == 0:
            initial_state += initial_state_per_node_seed
        else:
            initial_state += initial_state_per_node

    # Model equations
    equations = []
    cost = 0
    for dest_node in range(num_nodes):
        s1 = s_list[dest_node]
        i1 = i_list[dest_node]
        if cull_vs_thin:
            cull = control_list[2*dest_node]
            thin = control_list[2*dest_node+1]
            s_to_r = thin * s1
        else:
            cull = control_list[dest_node]
            s_to_r = 0
        s_to_i = 0
        i_to_r = gamma * i1 + cull * i1
        for source_node in range(num_nodes):
            beta12 = betas[source_node, dest_node]
            i2 = i_list[source_node]
            s_to_i += s1 * i2 * beta12
        equations += [-s_to_i - s_to_r, s_to_i - i_to_r]
        # Objective term
        if not final_reward_only:
            cost -= s1

    if cull_vs_thin:
        # Overriding to 0 to align to target approach (too awkward to calc as a general thing).
        control_initial_point_per_step = [0.0, 0.0] * num_nodes
        control_bounds_lower = [0.0] * num_nodes * 2
        control_bounds_upper = [1.0] * num_nodes * 2
    else:
        # Overriding to 0 to align to target approach (too awkward to calc as a general thing).
        control_initial_point_per_step = [0.0] * num_nodes
        control_bounds_lower = [0]*num_nodes
        control_bounds_upper = [1.0]*num_nodes

    if cull_vs_thin:
        def inequality_constraint(state, control):
            b_sum = 0
            for node_idx in range(num_nodes):
                s_local = state[node_idx * 2]
                i_local = state[node_idx * 2 + 1]
                cull_local = control[2 * node_idx]
                thin_local = control[2 * node_idx + 1]
                b_sum += cull_local * cost_per_action * i_local
                b_sum += thin_local * cost_per_action * s_local
            return [(b_sum, 0, budget)]
    else:
        def inequality_constraint(state, control):
            b_sum = 0
            for node_idx in range(num_nodes):
                i_local = state[node_idx * 2 + 1]
                b_sum += control[node_idx] * i_local
            return [(b_sum, 0, budget)]
    state_bounds_upper = state_bounds_upper_per_node * num_nodes
    state_bounds_lower = state_bounds_lower_per_node * num_nodes

    if final_reward_only:
        def final_cost(state):
            cost_local = 0
            for node_idx in range(num_nodes):
                s_local = state[node_idx * 2]
                cost_local -= s_local
            return cost_local
    else:
        def final_cost(_):
            return 0

    casadi_inputs = {'time_horizon': horizon,
                     'control_intervals': steps,
                     'ode_resolution_multiplier': 6,
                     'integrator_type': 'cvodes',
                     'use_midpoint': True,
                     'state_list': state_list,
                     'control_list': control_list,
                     'dynamics_list': equations,
                     'objective': cost,
                     'final_cost': final_cost,
                     'initial_state': initial_state,
                     'control_bounds_lower': control_bounds_lower,
                     'control_bounds_upper': control_bounds_upper,
                     'inequality_constraint_function': inequality_constraint,
                     'state_bounds_lower': state_bounds_lower,
                     'state_bounds_upper': state_bounds_upper,
                     'control_initial_trajectories': [control_initial_point_per_step] * steps}

    line_types = (control_line_type, state_line_type)

    return casadi_inputs, line_types


def run_multi_node(num_nodes,
                   plot=False,
                   node_population=1.0,
                   cull_vs_thin=False,
                   horizon=20,
                   steps=20):
    casadi_inputs, line_types = generate_multi_node(num_nodes,
                                                    node_population,
                                                    cull_vs_thin,
                                                    horizon=horizon,
                                                    steps=steps)

    opt = OptimalControlCollocation(**casadi_inputs)
    success, u_opt, x_opt = opt.run()

    if plot:
        control_line_type, state_line_type = line_types
        opt.plot(state_line_type, control_line_type)
    return success, u_opt


def test_multi_node_optimal_control():
    success, u_opt = run_multi_node(8, plot=False)
    assert success
    assert (len(u_opt[0]) == 20)


def test_cull_vs_thin_optimal_control():
    success, u_opt = run_multi_node(4, plot=False, cull_vs_thin=True)
    assert success
    assert (len(u_opt[0]) == 20)
