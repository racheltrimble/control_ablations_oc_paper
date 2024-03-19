# Basing on code from here:
# https://github.com/casadi/casadi/blob/master/docs/examples/python/direct_single_shooting.py
import numpy as np
import casadi as ca
from matplotlib import pyplot as plt

from control_ablations.generic_targets.integrators import get_cvodes, get_rk4, get_midpoint
from control_ablations import config

class OptimalControl:
    def __init__(self,
                 time_horizon,
                 control_intervals,
                 state_list,
                 control_list,
                 dynamics_list,
                 objective,
                 final_cost,
                 initial_state,
                 control_bounds_lower,
                 control_bounds_upper,
                 inequality_constraint_function,
                 integrator_type,
                 control_initial_trajectories,
                 check_initial_control=False):

        self.integrator_type = integrator_type
        self.time_horizon = time_horizon
        self.control_intervals = control_intervals
        self.timestep_duration = self.time_horizon / self.control_intervals / self.get_ode_resolution_multiplier()
        print(f"Using timestep of {self.timestep_duration}")
        self.state = ca.vertcat(*state_list)
        self.state_labels = []
        for state in state_list:
            self.state_labels += [state.name()]

        if len(control_list) == 1:
            self.control_vector = control_list[0]
        else:
            self.control_vector = ca.vertcat(*control_list)

        self.control_labels = []
        for control in control_list:
            self.control_labels += [control.name()]

        self.dynamics = ca.vertcat(*dynamics_list)
        self.objective = objective
        if final_cost is None:
            self.use_final_cost = False
        else:
            self.use_final_cost = True
            answer = final_cost(self.state)
            self.final_cost = ca.Function("final_cost", [self.state], [answer])
        self.initial_state = initial_state
        self.control_bounds = (control_bounds_lower, control_bounds_upper)
        if control_initial_trajectories is None:
            default_control = list((np.array(control_bounds_lower) + np.array(control_bounds_upper)) / 2)
            self.control_initial_trajectories = [default_control] * self.control_intervals
        else:
            self.control_initial_trajectories = control_initial_trajectories

        self.inequality_constraint_function = inequality_constraint_function

        # Check provided initial control complies to the constraints.
        # This defaults to not run because in the case of iterative optimal control (MPC), it is too sensitive to
        # numerical issues.
        if check_initial_control:
            self.check_control_against_constraints(self.control_initial_trajectories[0], self.initial_state)

        # Initialise empty NLP
        # Start with an empty NLP
        self.optimiser_input_variables_per_timestep = []
        self.optimiser_input_initialisation_per_timestep = []
        self.optimiser_input_lower_bounds_per_timestep = []
        self.optimiser_input_upper_bounds_per_timestep = []
        self.reward_accumulator = 0

        # inequality bounds
        self.g = []
        self.lbg = []
        self.ubg = []

        # placeholder for the solution
        self.solution = None

    def check_control_against_constraints(self, control_point, state):
        control_bounds_lower, control_bounds_upper = self.control_bounds
        assert (np.logical_or(np.array(control_point) <= np.array(control_bounds_upper),
                              np.isclose(control_point, control_bounds_upper))).all()
        assert (np.logical_or(np.array(control_point) >= np.array(control_bounds_lower),
                              np.isclose(control_point, control_bounds_lower, atol=5e-8)).all())
        self.check_inequality_constraints(state, control_point)

    def check_inequality_constraints(self, state, control_point):
        # Check provided initial state and control complies to the inequality constraints.
        if self.inequality_constraint_function is not None:
            s = ca.SX.sym('state', len(self.initial_state))
            c = ca.SX.sym('control', len(self.control_initial_trajectories[0]))
            test_constraints_list = self.inequality_constraint_function(s, c)

            for constraint in test_constraints_list:
                constraint_eq, lower_bound, upper_bound = constraint
                test_func = ca.Function('test', [s, c], [constraint_eq])
                initial_constraint_point = test_func(state, control_point)
                assert (initial_constraint_point < upper_bound or
                        np.isclose(initial_constraint_point, upper_bound, atol=1e-7))
                assert (initial_constraint_point > lower_bound or
                        np.isclose(initial_constraint_point, lower_bound, atol=1e-7))

    def solve(self):
        # Check inputs are free from NaNs.
        assert (not np.isnan(self.optimiser_input_initialisation_per_timestep).any())
        # Create an NLP solver
        prob = {'f': self.reward_accumulator,
                'x': ca.vertcat(*self.optimiser_input_variables_per_timestep),
                'g': ca.vertcat(*self.g)}

        # According to this https://web.casadi.org/python-api/#nlp
        # Requires ma97 solver installed as per https://github.com/casadi/casadi/wiki/Obtaining-HSL
        # Edited a few lines in the configure tool to avoid version flags causing issues with more recent GCC versions.
        # Reinstalled command line tools to avoid issue with linker.
        opts = {'ipopt': {'fixed_variable_treatment': 'make_constraint',
                          'linear_solver': 'ma97',
                          'ma97_scaling': 'none',
                          'ma97_order': 'metis',
                          'check_derivatives_for_naninf': 'yes',
                          'print_user_options': 'yes',
                          'tol': 1e-6,
                          'mu_strategy': "adaptive",
                          'hsllib': config.hsl_path,
                          # 'derivative_test': 'second-order'
                          },
                'inputs_check': True,
                'regularity_check': True,
                'warn_initial_bounds': True}

        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        for x0, upper, lower in zip(self.optimiser_input_initialisation_per_timestep,
                                    self.optimiser_input_upper_bounds_per_timestep,
                                    self.optimiser_input_lower_bounds_per_timestep):
            assert (x0 <= upper or np.isclose(x0, upper))
            assert (x0 >= lower or np.isclose(x0, lower))

        # Solve the NLP
        self.solution = solver(x0=self.optimiser_input_initialisation_per_timestep,
                               lbx=self.optimiser_input_lower_bounds_per_timestep,
                               ubx=self.optimiser_input_upper_bounds_per_timestep,
                               lbg=self.lbg,
                               ubg=self.ubg)
        stats = solver.stats()
        if stats['return_status'] == 'Infeasible_Problem_Detected':
            self.debug_final_point()
            # Assumes fail is caught elsewhere...
        print(stats)
        return stats['success']

    def debug_final_point(self):
        print("Debugging failed output from optimiser")
        # Do x and u meet original constraints?
        x_opt, u_opt = self.get_trajectories()
        per_step_x = x_opt.transpose()
        per_step_u = u_opt.transpose()
        ode_res = self.get_ode_resolution_multiplier()
        for control_step in range(per_step_u.shape[0]):
            for step in range(control_step * ode_res, (control_step + 1) * ode_res):
                self.check_control_against_constraints(per_step_u[control_step], per_step_x[step])

        # What about the detailed constraints?
        output_constraints = self.solution['g'].full()[:, 0]
        where_lower_broken = np.where(output_constraints < self.lbg)[0]
        where_upper_broken = np.where(output_constraints > self.ubg)[0]
        flat_g = ca.vertcat(*self.g)
        print("Lower:")
        for index in where_lower_broken[0:100]:
            print(f"Lower - {index}: {flat_g[index]} ({self.lbg[index]})")
        print("Upper:")
        for index in where_upper_broken[0:100]:
            print(f"Upper - {index}: {flat_g[index]} ({self.ubg[index]})")

    def get_jacobian(self):
        # Debug sanity check for jacobian
        return ca.jacobian(ca.vertcat(*self.g), ca.vertcat(*self.optimiser_input_variables_per_timestep))

    # https://sourceforge.net/p/casadi/discussion/1271244/thread/c6df4d27/
    def get_hessian_nnz(self):
        # Debug sanity check for hessian of the constraints.
        # (assumes hessian of objective is zero)
        # Goes through each constraint and works out the hessian in turn and sums
        # i.e. as if all lambdas were 1.
        flattened_constraints = ca.vertcat(*self.g)
        hessian_1 = None
        for constraint_idx in range(flattened_constraints.shape[0]):
            constraint = flattened_constraints[constraint_idx]
            hessian, grad = ca.hessian(constraint, ca.vertcat(*self.optimiser_input_variables_per_timestep))
            if hessian_1 is None:
                hessian_1 = hessian
            else:
                hessian_1 += hessian
        nnz = hessian_1.nnz()
        return nnz

    def add_per_timestep_control_and_bounds(self, per_step_state, timestep):
        per_step_control = []
        for control_index in range(self.control_vector.shape[0]):
            per_step_control += [ca.MX.sym(f'control_{str(self.control_labels[control_index])}_t{str(timestep)}')]
        if self.inequality_constraint_function is not None:
            constraints_list = self.inequality_constraint_function(per_step_state, per_step_control)
            for constraint in constraints_list:
                constraint_eq, lower_bound, upper_bound = constraint
                self.g += [constraint_eq]
                self.lbg += [lower_bound]
                self.ubg += [upper_bound]

        self.optimiser_input_initialisation_per_timestep += self.control_initial_trajectories[timestep]
        new_control_lower_bound, new_control_upper_bound = self.control_bounds
        self.optimiser_input_variables_per_timestep += per_step_control
        self.optimiser_input_lower_bounds_per_timestep += new_control_lower_bound
        self.optimiser_input_upper_bounds_per_timestep += new_control_upper_bound
        if len(per_step_control) == 1:
            per_step_control = per_step_control[0]
        else:
            per_step_control = ca.vertcat(*per_step_control)
        return per_step_control

    def plot(self,
             state_line_type,
             control_line_type,
             show_state_plots=None,
             show_control_plots=None,
             filepath=None):
        assert self.solution is not None
        ode_resolution_multiplier = self.get_ode_resolution_multiplier()
        state_trajectories, controls = self.get_trajectories()
        self.plot_from_given_state_and_control(state_trajectories,
                                               controls,
                                               state_line_type,
                                               control_line_type,
                                               show_state_plots,
                                               show_control_plots,
                                               filepath,
                                               ode_resolution_multiplier=ode_resolution_multiplier)

    def get_ode_resolution_multiplier(self):
        return 1

    def plot_from_given_state_and_control(self,
                                          state_trajectories,
                                          controls,
                                          state_line_type,
                                          control_line_type,
                                          show_state_plots=None,
                                          show_control_plots=None,
                                          filepath=None,
                                          ode_resolution_multiplier=1):
        if show_state_plots is None:
            show_state_plots = [True] * len(self.initial_state)
        if show_control_plots is None:
            show_control_plots = [True] * self.control_vector.shape[0]

        assert self.state_labels is not None
        assert self.control_labels is not None

        grid_timesteps = self.control_intervals * ode_resolution_multiplier

        time_grid = [self.time_horizon / grid_timesteps * k for k in range(grid_timesteps + 1)]
        control_time_grid = [self.time_horizon / self.control_intervals * k for k in range(self.control_intervals)]
        fig = plt.figure(1)
        plt.clf()
        for index, trajectory in enumerate(state_trajectories):
            if show_state_plots[index]:
                plt.plot(time_grid, trajectory, state_line_type[index], label=self.state_labels[index])

        for index, control in enumerate(controls):
            if show_control_plots[index]:
                label = self.control_labels[index]
                line_type = control_line_type[index]
                plt.step(control_time_grid,
                         control,
                         line_type,
                         label=label,
                         alpha=0.5)
        plt.xlabel('t')
        plt.title("Optimal control level plot")
        plt.legend()
        plt.grid()
        if filepath is None:
            plt.show()
        else:
            parent = filepath.parents[0]
            if not parent.exists():
                parent.mkdir(parents=True)
            fig.savefig(filepath)
            plt.close(fig)

    def get_trajectories(self):
        raise NotImplementedError

    def get_integrator(self, ode_resolution_multiplier=1):
        steps_per_interval = 1
        if self.integrator_type == "cvodes":
            dynamics_integrator = get_cvodes(self.state,
                                             self.control_vector,
                                             self.dynamics,
                                             self.objective,
                                             self.time_horizon / self.control_intervals / ode_resolution_multiplier)
        elif self.integrator_type == "rk4":
            dynamics_integrator = get_rk4(self.state,
                                          self.control_vector,
                                          self.dynamics,
                                          self.objective,
                                          self.time_horizon / self.control_intervals / ode_resolution_multiplier,
                                          steps_per_interval)
        elif self.integrator_type == "midpoint":
            dynamics_integrator = get_midpoint(self.state,
                                               self.control_vector,
                                               self.dynamics,
                                               self.objective,
                                               self.time_horizon / self.control_intervals / ode_resolution_multiplier,
                                               steps_per_interval)
        else:
            print("Unsupported integrator type requested")
            assert 0
        return dynamics_integrator

    def get_integrated_trajectories(self,
                                    control,
                                    ode_resolution_multiplier=1,
                                    state_bounds_upper=None,
                                    state_bounds_lower=None):
        # If state bounds are available, they can be used to minimise runaway errors
        state_trajectory = [np.array(self.initial_state)]
        control = np.array(control).transpose()
        assert control.shape == (len(self.control_labels), self.control_intervals)
        dynamics_integrator = self.get_integrator(ode_resolution_multiplier=ode_resolution_multiplier)
        for control_step in range(self.control_intervals):
            for ode_step in range(ode_resolution_multiplier):
                integrator_output = dynamics_integrator(x0=state_trajectory[-1],
                                                        p=control[:, control_step])
                integrator_data = integrator_output['xf'].full()

                if np.isnan(integrator_data).any():
                    print("found NaN elements")
                assert (not np.isnan(integrator_data).any())
                new_elements = integrator_data[:, 0]
                if state_bounds_upper is not None:
                    new_elements = np.minimum(new_elements, state_bounds_upper)
                if state_bounds_lower is not None:
                    new_elements = np.maximum(new_elements, state_bounds_lower)
                state_trajectory += [new_elements]

        state_trajectory = np.array(state_trajectory)
        state_trajectories = []
        for index in range(len(self.initial_state)):
            state_trajectories += [ca.vcat([r[index] for r in state_trajectory])]

        num_control = self.control_vector.shape[0]
        controls = []
        for index in range(num_control):
            raw_control = control[index::num_control]
            if isinstance(raw_control, np.ndarray):
                if len(raw_control.shape) == 2:
                    raw_control = raw_control[0]
            controls += [raw_control]

        return state_trajectories, controls


class OptimalControlDirect(OptimalControl):
    def __init__(self,
                 time_horizon,
                 control_intervals,
                 state_list,
                 control_list,
                 dynamics_list,
                 objective,
                 final_cost,
                 initial_state,
                 control_bounds_lower,
                 control_bounds_upper,
                 inequality_constraint_function,
                 integrator_type="rk4",
                 control_initial_trajectories=None):
        super().__init__(time_horizon,
                         control_intervals,
                         state_list,
                         control_list,
                         dynamics_list,
                         objective,
                         final_cost,
                         initial_state,
                         control_bounds_lower,
                         control_bounds_upper,
                         inequality_constraint_function,
                         integrator_type,
                         control_initial_trajectories)
        self.use_rk4 = False

    def run(self):
        self.generate_nlp()
        success = self.solve()
        u_opt = self.get_trajectories()
        x_opt = self.get_integrated_trajectories(u_opt)
        return success, u_opt[:, 1:], x_opt[:, 1:]

    def dynamics_sanity(self, start_state, control, iterations=1, plot=False):
        self.get_integrator()
        state = start_state
        state_trajectory = [start_state]
        dynamics_integrator = self.get_integrator()
        for i in range(iterations):
            output = dynamics_integrator(x0=state, p=control)
            print("Iteration: ", i)
            print(output['xf'].full())
            print(output['qf'])
            state = output['xf'].full()
            state_trajectory += [state]

        if plot:
            time_grid = [self.time_horizon / self.control_intervals * k for k in range(iterations + 1)]
            for index in range(len(self.initial_state)):
                trajectory = ca.vcat([r[index] for r in state_trajectory])
                plt.plot(time_grid, trajectory, label=self.state_labels[index])
        plt.show()

    def generate_nlp(self):
        per_step_state = ca.MX(self.initial_state)
        dynamics_integrator = self.get_integrator()
        for timestep in range(self.control_intervals):
            # New NLP variable for the control
            per_step_control = self.add_per_timestep_control_and_bounds(per_step_state, timestep)
            integrator_output = dynamics_integrator(x0=per_step_state, p=per_step_control)
            per_step_state = integrator_output['xf']
            if not self.use_final_cost:
                self.reward_accumulator += integrator_output['qf']
        if self.use_final_cost:
            self.reward_accumulator += self.final_cost(per_step_state)

    def get_trajectories(self):
        return self.get_integrated_trajectories(self.solution['x'])


class OptimalControlCollocation(OptimalControl):
    def __init__(self,
                 time_horizon,
                 control_intervals,
                 state_list,
                 control_list,
                 dynamics_list,
                 objective,
                 final_cost,
                 initial_state,
                 control_bounds_lower,
                 control_bounds_upper,
                 inequality_constraint_function,
                 integrator_type,
                 state_bounds_lower,
                 state_bounds_upper,
                 polynomial_degree=2,
                 use_midpoint=True,
                 control_initial_trajectories=None,
                 initialisation_plot_path=None,
                 ode_resolution_multiplier=1,
                 mpc_iteration=False
                 ):
        self.ode_resolution_multiplier = ode_resolution_multiplier
        super().__init__(time_horizon,
                         control_intervals,
                         state_list,
                         control_list,
                         dynamics_list,
                         objective,
                         final_cost,
                         initial_state,
                         control_bounds_lower,
                         control_bounds_upper,
                         inequality_constraint_function,
                         integrator_type,
                         control_initial_trajectories)

        # perform basic enforcement on control bounds
        if mpc_iteration:
            self.enforce_control_bounds()

        # Colocation feeds intermediate states in as variables to be solver
        # hence, the solver needs bounds.
        self.state_bounds_lower = state_bounds_lower
        self.state_bounds_upper = state_bounds_upper
        self.state_initalisation_trajectory, controls = \
            self.get_integrated_trajectories(self.control_initial_trajectories,
                                             ode_resolution_multiplier,
                                             state_bounds_upper=state_bounds_upper,
                                             state_bounds_lower=state_bounds_lower)
        trajectory_matrix = ca.horzcat(*self.state_initalisation_trajectory)
        for timestep in range(trajectory_matrix.shape[0]):
            state = trajectory_matrix[timestep, :]
            lower_violation = np.array(state - ca.horzcat(*state_bounds_lower))
            upper_violation = np.array(state - ca.horzcat(*state_bounds_upper))
            if not (lower_violation >= 0).all():
                # If it's a proper violation, and we're not doing MPC then stop
                assert mpc_iteration or (np.logical_or(lower_violation >= 0, np.isclose(lower_violation, 0))).all()
                # If it's just floating point faff then fix to avoid firing later warnings in casadi.
                offending_indices = np.where(lower_violation < 0)[1]
                for state_idx in offending_indices:
                    self.state_initalisation_trajectory[state_idx][timestep] = state_bounds_lower[state_idx]
            if not (upper_violation <= 0).all():
                assert (np.logical_or(upper_violation <= 0, np.isclose(upper_violation, 0))).all()
                offending_indices = np.where(upper_violation > 0)[1]
                for state_idx in offending_indices:
                    self.state_initalisation_trajectory[state_idx][timestep] = state_bounds_upper[state_idx]

        # Check generated initial state and control complies to the inequality constraints.
        if not mpc_iteration:
            for idx in range(trajectory_matrix.shape[0] - 1):
                control_index = int(idx/self.ode_resolution_multiplier)
                self.check_inequality_constraints(trajectory_matrix[idx, :],
                                                  self.control_initial_trajectories[control_index])

        assert (not np.isnan(self.state_initalisation_trajectory).any())
        if initialisation_plot_path is not None:
            self.plot_from_given_state_and_control(self.state_initalisation_trajectory,
                                                   controls,
                                                   state_line_type=len(state_list) * ['.'],
                                                   control_line_type=len(control_list) * ['.-'],
                                                   filepath=initialisation_plot_path,
                                                   ode_resolution_multiplier=ode_resolution_multiplier
                                                   )

        self.polynomial_degree = polynomial_degree
        self.use_midpoint = use_midpoint
        self.collocation_derivative_equation_coefficients = None
        self.polynomial_coefficients_end_of_timestep_per_yval = None
        self.reward_function_coefficients = None

        # Optimal trajectories are generated as part of the optimisation
        self.x_plot = None
        self.u_plot = None

    def enforce_control_bounds(self):
        lower = self.control_bounds[0]
        upper = self.control_bounds[1]
        for idx, timestep in enumerate(self.control_initial_trajectories):
            control_array = np.array(timestep)
            control_array = np.minimum(control_array, upper)
            control_array = np.maximum(control_array, lower)
            self.control_initial_trajectories[idx] = list(control_array)

    def run(self):
        if not self.use_midpoint:
            assert self.ode_resolution_multiplier == 1
        if self.use_midpoint:
            self.generate_midpoint_nlp()
        else:
            self.setup_collocation()
            self.generate_nlp()
        success = self.solve()
        x_opt, u_opt = self.get_trajectories()

        for control_point in range(self.control_intervals):
            state_point = control_point * self.get_ode_resolution_multiplier()
            self.check_control_against_constraints(u_opt[:, control_point], x_opt[:, state_point])

        return success, u_opt, x_opt

    def setup_collocation(self):
        # Get collocation points
        collocation_points = np.append(0, ca.collocation_points(self.polynomial_degree, 'legendre'))
        print("Collocation points:", collocation_points)

        self.collocation_derivative_equation_coefficients = \
            np.zeros((self.polynomial_degree + 1, self.polynomial_degree + 1))
        self.polynomial_coefficients_end_of_timestep_per_yval = np.zeros(self.polynomial_degree + 1)
        self.reward_function_coefficients = np.zeros(self.polynomial_degree + 1)

        # Construct polynomial basis
        # Lagrange polynomial is an equation describing the minimal polynomial which passes through n points.
        # https://mathworld.wolfram.com/LagrangeInterpolatingPolynomial.html
        # This maps between the y values of the given points and the polynomial coefficients for the
        # equation describing y in terms of the fraction of the timestep which has elapsed.
        for y_val_index in range(self.polynomial_degree + 1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            lagrange_polynomials = np.poly1d([1])
            for point_index in range(self.polynomial_degree + 1):
                if point_index != y_val_index:
                    new_poly = np.poly1d([1, -collocation_points[point_index]])
                    divisor = (collocation_points[y_val_index] - collocation_points[point_index])
                    lagrange_polynomials *= new_poly / divisor

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            self.polynomial_coefficients_end_of_timestep_per_yval[y_val_index] = lagrange_polynomials(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points
            # to get the coefficients of the continuity equation
            pder = np.polyder(lagrange_polynomials)
            for point_index in range(self.polynomial_degree + 1):
                self.collocation_derivative_equation_coefficients[y_val_index, point_index] = \
                    pder(collocation_points[point_index])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(lagrange_polynomials)
            self.reward_function_coefficients[y_val_index] = pint(1.0)

    def generate_nlp(self):
        # For plotting x and u given w
        x_plot = []
        u_plot = []

        # Define initial conditions as constraints
        num_states = len(self.initial_state)
        per_step_state = np.array(self.initial_state)
        x_plot.append(per_step_state)

        # Continuous time dynamics
        f = ca.Function('f', [self.state, self.control_vector], [self.dynamics, self.objective])
        # Formulate the NLP
        trajectory_matrix = ca.horzcat(*self.state_initalisation_trajectory)
        for timestep in range(self.control_intervals):
            per_step_control = self.add_per_timestep_control_and_bounds(per_step_state, timestep)
            u_plot.append(per_step_control)
            state_initialisation = trajectory_matrix[timestep, :].elements()

            # State at collocation points
            per_collocation_point_state = []
            for point_index in range(self.polynomial_degree):
                new_state = ca.MX.sym('X_' + str(timestep) + '_' + str(point_index), len(self.initial_state))
                per_collocation_point_state.append(new_state)
                self.optimiser_input_variables_per_timestep += [new_state]
                self.optimiser_input_lower_bounds_per_timestep += self.state_bounds_lower
                self.optimiser_input_upper_bounds_per_timestep += self.state_bounds_upper
                self.optimiser_input_initialisation_per_timestep += list(state_initialisation)

            # The contribution to the state at the end of the timestep from the state at the start.
            state_at_end_of_timestep = self.polynomial_coefficients_end_of_timestep_per_yval[0] * per_step_state

            # Loop over collocation points
            for point_index in range(1, self.polynomial_degree + 1):
                # Expression for the state derivative at the collocation point
                xp = self.collocation_derivative_equation_coefficients[0, point_index] * per_step_state
                # Sum up the derivative of the Lagrange interpolating polynomial.
                for y_val_index in range(self.polynomial_degree):
                    d_coefficients = \
                        self.collocation_derivative_equation_coefficients[y_val_index + 1, point_index]
                    xp += d_coefficients * per_collocation_point_state[y_val_index]

                # Append collocation equations
                this_state = per_collocation_point_state[point_index - 1]
                gradient_to_match, cost_at_collocation = f(this_state, per_step_control)
                # xp derivatives are in terms of progress within the timestep. By chain rule, just divide
                # by the timestep to rebase to "real" time
                self.g += [self.timestep_duration * gradient_to_match - xp]
                self.lbg += [0] * num_states
                self.ubg += [0] * num_states

                # Add contribution to the end state
                state_at_end_of_timestep += \
                    self.polynomial_coefficients_end_of_timestep_per_yval[point_index] * this_state

                # Add contribution to reward
                new_contribution = self.reward_function_coefficients[point_index] * cost_at_collocation
                # Reward has been integrated wrt progress within timestep so must be multiplied by the timestep.
                if not self.use_final_cost:
                    self.reward_accumulator += new_contribution * self.timestep_duration

            # New NLP variable for state at end of interval
            per_step_state = ca.MX.sym('X_' + str(timestep + 1), num_states)
            self.optimiser_input_variables_per_timestep += [per_step_state]
            self.optimiser_input_lower_bounds_per_timestep += self.state_bounds_lower
            self.optimiser_input_upper_bounds_per_timestep += self.state_bounds_upper
            self.optimiser_input_initialisation_per_timestep += state_initialisation
            x_plot.append(per_step_state)

            # Add equality constraint
            self.g += [state_at_end_of_timestep - per_step_state]
            self.lbg += [0] * num_states
            self.ubg += [0] * num_states

        if self.use_final_cost:
            self.reward_accumulator += self.final_cost(per_step_state)
        # u_plot.append([ca.vertcat(*[np.nan] * len(self.control_labels))])
        # Concatenate vectors
        self.x_plot = ca.horzcat(*x_plot)
        self.u_plot = ca.horzcat(*u_plot)

    def generate_midpoint_nlp(self):
        # For plotting x and u given w
        x_plot = []
        u_plot = []

        # Define initial conditions as constraints
        num_states = len(self.initial_state)
        per_step_state = np.array(self.initial_state)
        x_plot.append(per_step_state)

        # Continuous time dynamics
        f = ca.Function('f', [self.state, self.control_vector], [self.dynamics, self.objective])

        # Formulate the NLP
        trajectory_matrix = ca.horzcat(*self.state_initalisation_trajectory)
        for control_timestep in range(self.control_intervals):
            per_step_control = self.add_per_timestep_control_and_bounds(per_step_state, control_timestep)
            u_plot.append(per_step_control)
            for state_timestep in range(self.ode_resolution_multiplier):
                state_initialisation = trajectory_matrix[control_timestep, :].elements()

                # New NLP variable for state at end of interval
                next_step_state = ca.MX.sym('X_' + str(control_timestep + 1), num_states)
                self.optimiser_input_variables_per_timestep += [next_step_state]
                self.optimiser_input_lower_bounds_per_timestep += self.state_bounds_lower
                self.optimiser_input_upper_bounds_per_timestep += self.state_bounds_upper
                self.optimiser_input_initialisation_per_timestep += state_initialisation
                x_plot.append(next_step_state)
                # State at midpoint and at end
                midpoint_state = (per_step_state + next_step_state) / 2

                # Match gradient at midpoint (collocation equations)
                gradient_to_match, cost_at_midpoint = f(midpoint_state, per_step_control)

                # Add contribution to reward
                if not self.use_final_cost:
                    self.reward_accumulator += cost_at_midpoint

                # Constrain state at end of timestep in terms of start and gradient
                continuity_constraint = per_step_state + self.timestep_duration * gradient_to_match
                self.g += [next_step_state - continuity_constraint]
                self.lbg += [0] * num_states
                self.ubg += [0] * num_states

                per_step_state = next_step_state

        if self.use_final_cost:
            self.reward_accumulator += self.final_cost(next_step_state)

        # Concatenate vectors
        self.x_plot = ca.horzcat(*x_plot)
        self.u_plot = ca.horzcat(*u_plot)

    def get_trajectories(self):
        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories',
                                   [ca.vertcat(*self.optimiser_input_variables_per_timestep)],
                                   [self.x_plot, self.u_plot], ['w'], ['x', 'u'])

        x_opt, u_opt = trajectories(self.solution['x'])
        x_opt = x_opt.full()  # to numpy array
        u_opt = u_opt.full()  # to numpy array
        return x_opt, u_opt

    def get_ode_resolution_multiplier(self):
        return self.ode_resolution_multiplier
