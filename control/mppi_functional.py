import jax
import jax.numpy as jnp
from jax.random import multivariate_normal
from jax import jit, lax

from utils_3d import *
from robot_config import *

# if no gpu available
# jax.config.update('jax_platform_name', 'cpu')


    
def setup_mppi_controller(learned_CSDF = None, robot_n = 8, input_size = 8, initial_horizon=10, samples = 10, control_bound = 0.2, dt=0.05, u_guess=None, use_GPU=True, costs_lambda = 0.03,  cost_goal_coeff = 0.2, cost_safety_coeff = 10.0, cost_perturbation_coeff=0.1, cost_goal_coeff_final = 0.2, cost_safety_coeff_final = 10.0, cost_state_coeff = 10.0):

    """
    Set up the MPPI controller.

    :param learned_CSDF: Learned N-CSDF for robot shape modeling (a link)
    :param robot_n: Dimension of states
    :param input_size: Control input dimension. In the CDSR mode, it equals to robot_n
    :param horizon: Initial prediction time horizon
    :param samples: Number of control samples around the initial guess
    :param control_bound: Control bounds
    :param dt: Time discretization
    :param u_guess: Initial control sequence guess
    :param use_GPU: Flag to use GPU for computations
    :param costs_lambda: parameter for cost weighting
    :param cost_goal_coeff: Coefficient for goal reaching cost
    :param cost_safety_coeff: Coefficient for safety cost
    :param cost_perturbation_coeff: Coefficient for control perturbation cost
    :param cost_goal_coeff_final: Coefficient for goal reaching cost at the final step
    :param cost_safety_coeff_final: Coefficient for safety cost at the final step
    :param cost_state_coeff: Coefficient for state constraint violation cost

    :return: MPPI controller function

    """
    # self.key = jax.random.PRNGKey(111)
    horizon = initial_horizon
    samples = samples
    robot_m = input_size
    dt = dt
    use_gpu = use_GPU

    jax_params = learned_CSDF

    control_mu = jnp.zeros(robot_m) 
    control_cov = 0.1 * jnp.eye(robot_m) 
    control_cov_inv = jnp.linalg.inv(control_cov)
    control_bound = control_bound
    if u_guess != None:
        U = u_guess
    else:
        U = 0.1 * jnp.ones((horizon,robot_m))

    costs_lambda = costs_lambda
    cost_goal_coeff = cost_goal_coeff
    cost_safety_coeff = cost_safety_coeff
    cost_perturbation_coeff = cost_perturbation_coeff
    cost_goal_coeff_final = cost_goal_coeff_final
    cost_safety_coeff_final = cost_safety_coeff_final

    cost_state_coeff = cost_state_coeff


    @jit
    def robot_dynamics_step(state, input):
        # Continuum robot

        return state + input * dt
    
    @jit
    def weighted_sum(U, perturbation, costs):
        costs = costs - jnp.min(costs)
        costs = costs / jnp.max(costs)
        lambd = costs_lambda
        weights = jnp.exp( - 1.0/lambd * costs )   # higher cost -> higher weight
        normalization_factor = jnp.sum(weights)
        def body(i, inputs):
            U = inputs
            U = U + perturbation[i] * weights[i] / normalization_factor
            return U
        return lax.fori_loop( 0, samples, body, (U) )
    
    @jit
    def single_sample_rollout(goal, robot_states_init, perturbed_control, obstaclesX, perturbation, safety_margin = 0.1):
        # Initialize robot_state
        robot_states = jnp.zeros((robot_n, horizon))
        robot_states = robot_states.at[:,0].set(robot_states_init)

        # Import the state constraints
        nominal_length = LINK_LENGTH
        min_length = MIN_CABLE_LENGTH
        max_length = MAX_CABLE_LENGTH


        # if dealing with sphere objects (e.g, radius = 0.7), if running the main_sphere, uncomment the following line:
        # safety_margin = 0.1 + 0.7                     

        
        # loop over horizon
        cost_sample = 0
        def body(i, inputs):
            cost_sample, robot_states, obstaclesX = inputs

            # get robot state
            robot_state = robot_states[:,[i]]

            robot_state = robot_state.reshape(NUM_OF_LINKS, 2)


            end_center, _, _ = compute_end_circle(robot_state)

            # Compute the distance between the end center and the goal
            end_center_distance = jnp.linalg.norm(goal - end_center)

            #jax.debug.print("🤯 i {index} end_center_distance {x} 🤯, state {state},", index=i, x=end_center_distance, state=robot_state.reshape(1,-1))

            cost_sample = cost_sample + cost_goal_coeff * end_center_distance            
            cost_sample = cost_sample + cost_perturbation_coeff * ((perturbed_control[:, [i]]-perturbation[:,[i]]).T @ control_cov_inv @ perturbation[:,[i]])[0,0]

            '''
            obstacle avoidance cost
            '''

            csdf_distances = evaluate_model(jax_params, robot_state.squeeze(), obstaclesX)
            cost_sample = cost_sample + cost_safety_coeff / jnp.max(jnp.array([jnp.min(csdf_distances)- safety_margin, 0.01]))


            # Compute the state constraint violation cost
            length_1 = robot_state.squeeze()[:, 0]
            length_2 = robot_state.squeeze()[:, 1]
            length_3 = 3 * nominal_length - length_1 - length_2

            # Compute the state constraint violation for each edge length, including the third edge length
            state_constraint_violation_1 = jnp.maximum(min_length - length_1, 0) + jnp.maximum(length_1 - max_length, 0)
            state_constraint_violation_2 = jnp.maximum(min_length - length_2, 0) + jnp.maximum(length_2 - max_length, 0)
            state_constraint_violation_3 = jnp.maximum(min_length - length_3, 0) + jnp.maximum(length_3 - max_length, 0)


            # Sum up the state constraint violations for all edge lengths and all links
            state_constraint_violation = jnp.sum(state_constraint_violation_1 + state_constraint_violation_2 + state_constraint_violation_3)

            cost_sample = cost_sample + cost_state_coeff * state_constraint_violation


            # Update robot states
            robot_states = robot_states.at[:,i+1].set(robot_dynamics_step(robot_states[:,[i]], perturbed_control[:, [i]])[:,0])
            return cost_sample, robot_states, obstaclesX
        
        cost_sample, robot_states, _ = lax.fori_loop(0, horizon-1, body, (cost_sample, robot_states, obstaclesX))

        # robot_configs = jax.vmap(state_to_config, in_axes=(1, None, None))(robot_states, link_radius, nominal_length)

        '''
        batched version of safety cost
        '''
        # robot_states_reshaped = robot_states[:, :-1].reshape(NUM_OF_LINKS, 2, -1)
        # csdf_distances = evaluate_model(jax_params, robot_states_reshaped, obstaclesX)

        # #Calculate minimum distances across the obstacle points for each configuration in the batch
        # min_csdf_distances = jnp.min(csdf_distances, axis=-1)

        # #Compute safety cost based on the minimum distances
        # safety_costs = cost_safety_coeff / jnp.maximum(min_csdf_distances - safety_margin, 0.01)

        # #Sum up the safety costs for all configurations in the batch
        # cost_sample += jnp.sum(safety_costs)


        robot_state = robot_states[:,[horizon-1]]

        robot_state = robot_state.reshape(NUM_OF_LINKS, 2)

        # Compute the end center and normal for the final state

        end_center, _, _ = compute_end_circle(robot_state)

        # Compute the distance between the end center and the goal for the final state
        end_center_distance = jnp.linalg.norm(goal - end_center)

        cost_sample = cost_sample + cost_goal_coeff_final * end_center_distance

        cost_sample = cost_sample + cost_perturbation_coeff * ((perturbed_control[:, [horizon]]-perturbation[:,[horizon]]).T @ control_cov_inv @ perturbation[:,[horizon]])[0,0]

        csdf_distances = evaluate_model(jax_params, robot_state.squeeze(), obstaclesX)
        
        cost_sample = cost_sample + cost_safety_coeff_final / jnp.max(jnp.array([jnp.min(csdf_distances) - safety_margin, 0.01]))

        return cost_sample, robot_states




    @jit
    def rollout_states_foresee(robot_init_state, perturbed_control, goal, obstaclesX, perturbation, safety_margin):

        ##### Initialize               
        # Robot
        robot_states = jnp.zeros( (samples, robot_n, horizon) )
        robot_states = robot_states.at[:,:,0].set( jnp.tile( robot_init_state.T, (samples,1) ) )

        # Cost
        cost_total = jnp.zeros(samples)

        # Single sample rollout
        if use_gpu:
            @jit
            def body_sample(robot_states_init, perturbed_control_sample, perturbation_sample):
                cost_sample, robot_states_sample = single_sample_rollout(goal, robot_states_init, perturbed_control_sample.T, obstaclesX, perturbation_sample.T, safety_margin)
                return cost_sample, robot_states_sample
            batched_body_sample = jax.vmap( body_sample, in_axes=0 )
            # print('gpu used')
            cost_total, robot_states, = batched_body_sample( robot_states[:,:,0], perturbed_control, perturbation)
        else:
            @jit
            def body_samples(i, inputs):
                robot_states, cost_total, obstaclesX = inputs     

                # Get cost
                cost_sample, robot_states_sample = single_sample_rollout(goal, robot_states[i,:,0], perturbed_control[i,:,:].T, obstaclesX, perturbation[i,:,:].T, safety_margin)
                cost_total = cost_total.at[i].set( cost_sample )
                robot_states = robot_states.at[i,:,:].set( robot_states_sample )
                return robot_states, cost_total, obstaclesX  
            robot_states, cost_total, obstaclesX = lax.fori_loop( 0, samples, body_samples, (robot_states, cost_total, obstaclesX) )

        return robot_states, cost_total

    @jit
    def compute_perturbed_control(subkey, control_mu, control_cov, control_bound, U):
        perturbation = multivariate_normal( subkey, control_mu, control_cov, shape=( samples, horizon ) ) # K x T x nu 
  
        perturbation = jnp.clip( perturbation, -0.5, 0.5) #0.3
        perturbed_control = U + perturbation

        perturbed_control = jnp.clip( perturbed_control, -control_bound, control_bound )
        perturbation = perturbed_control - U
        return perturbation, perturbed_control
    
    # fori_loop runs faster than lax.scan, at least for small horizon (<30)
    
    @jit
    def rollout_control(init_state, actions):
        states = jnp.zeros((robot_n, horizon+1))
        states = states.at[:,0].set(init_state.reshape(-1,1)[:,0])
        def body(i, inputs):
            states = inputs
            states = states.at[:,i+1].set( robot_dynamics_step(states[:,[i]], actions[i,:].reshape(-1,1))[:,0] )
            return states
        states = lax.fori_loop(0, horizon, body, states)
        return states

    
    @jit
    def compute_rollout_costs( key, U, init_state, goal, obstaclesX, safety_margin):

        perturbation, perturbed_control = compute_perturbed_control(key, control_mu, control_cov, control_bound, U)

        sampled_robot_states, costs= rollout_states_foresee(init_state, perturbed_control, goal, obstaclesX, perturbation, safety_margin)

        U = weighted_sum( U, perturbation, costs)

        states_final = rollout_control(init_state, U)              
        action = U[0,:].reshape(-1,1)
        U = jnp.append(U[1:,:], U[[-1],:], axis=0)

        sampled_robot_states = sampled_robot_states.reshape(( robot_n*samples, horizon ))
   
        return sampled_robot_states, states_final, action, U
    
    return compute_rollout_costs
    

    
        