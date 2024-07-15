import jax
import jax.numpy as jnp
from jax.random import multivariate_normal
from jax import jit, lax

from utils_3d import *
from robot_config import *

# if no gpu available
# jax.config.update('jax_platform_name', 'cpu')

def setup_mppi_controller(learned_CSDF = None, robot_n = 8, horizon=10, samples = 10, input_size = 8, control_bound = 0.2, dt=0.05, u_guess=None, use_GPU=True, costs_lambda = 0.03,  cost_goal_coeff = 0.2, cost_safety_coeff = 10.0, cost_perturbation_coeff=0.1, cost_goal_coeff_final = 0.2, cost_safety_coeff_final = 10.0, cost_state_coeff = 10.0):

    '''
    net: learned N-CSDF for robot shape modeling
    robot_n: dimension of states
    horizon: time horizon of MPPI
    samples: number of control samples around the initial guess
    input_size: control input size
    control_bound: control bounds
    dt: time discretization
    u_guess: initial control sequence guess
    costs_lambda: ? 
    cost_goal_coeff: parameters for cost on reaching the goal
    
    '''
    # self.key = jax.random.PRNGKey(111)
    horizon = horizon
    samples = samples
    robot_m = input_size
    dt = dt
    use_gpu = use_GPU

    jax_params = learned_CSDF

    control_mu = jnp.zeros(robot_m) 
    control_cov = 0.1 * jnp.eye(robot_m) #0.1 * jnp.eye(robot_m)
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
        # 2D Continuum robot

        return state + input * dt
    
    @jit
    def weighted_sum(U, perturbation, costs):#weights):
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
    def single_sample_rollout(goal, robot_states_init, perturbed_control, obstaclesX, perturbation):
        # Initialize robot_state
        robot_states = jnp.zeros((robot_n, horizon))
        robot_states = robot_states.at[:,0].set(robot_states_init)

        # Define the state constraints
        nominal_length = LINK_LENGTH

        min_length = 0.8 * nominal_length
        max_length = 1.2 * nominal_length
        link_radius = LINK_RADIUS

        # 0.13 is a safety margin
        safety_margin = 0.1                          #for the sphere env: 0.6 + 0.2

        
        # loop over horizon
        cost_sample = 0
        def body(i, inputs):
            cost_sample, robot_states, obstaclesX = inputs

            # get robot state
            robot_state = robot_states[:,[i]]

            robot_state = robot_state.reshape(NUM_OF_LINKS, 2)


            end_center, end_normal, _ = compute_end_circle(robot_state, link_radius, nominal_length)

            # Compute the distance between the end center and the goal
            end_center_distance = jnp.linalg.norm(goal - end_center)

            #jax.debug.print("🤯 i {index} end_center_distance {x} 🤯, state {state},", index=i, x=end_center_distance, state=robot_state.reshape(1,-1))

            cost_sample = cost_sample + cost_goal_coeff * end_center_distance            
            cost_sample = cost_sample + cost_perturbation_coeff * ((perturbed_control[:, [i]]-perturbation[:,[i]]).T @ control_cov_inv @ perturbation[:,[i]])[0,0]

            robot_config = state_to_config(robot_state.squeeze(), link_radius, nominal_length)

            csdf_distances = evaluate_model(jax_params, robot_config, robot_state.squeeze(), link_radius, nominal_length, obstaclesX)

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


        robot_state = robot_states[:,[horizon-1]]

        robot_state = robot_state.reshape(NUM_OF_LINKS, 2)

        # Compute the end center and normal for the final state

        end_center, _, _ = compute_end_circle(robot_state, link_radius, nominal_length)

        # Compute the distance between the end center and the goal for the final state
        end_center_distance = jnp.linalg.norm(goal - end_center)

        cost_sample = cost_sample + cost_goal_coeff_final * end_center_distance

        cost_sample = cost_sample + cost_perturbation_coeff * ((perturbed_control[:, [horizon]]-perturbation[:,[horizon]]).T @ control_cov_inv @ perturbation[:,[horizon]])[0,0]
        
        robot_config = state_to_config(robot_state.squeeze(), link_radius, nominal_length)


        csdf_distances = evaluate_model(jax_params, robot_config, robot_state.squeeze(), link_radius, nominal_length, obstaclesX)
        
        cost_sample = cost_sample + cost_safety_coeff_final / jnp.max(jnp.array([jnp.min(csdf_distances) - safety_margin, 0.01]))

        return cost_sample, robot_states

    # @jit
    # def single_sample_rollout(goal, robot_states_init, perturbed_control, obstaclesX, perturbation):
    #     # Initialize robot_state
    #     robot_states = jnp.zeros((robot_n, horizon))
    #     robot_states = robot_states.at[:,0].set(robot_states_init)

    #     # Define the state constraints
    #     nominal_length = LINK_LENGTH

    #     min_length = 0.8 * nominal_length
    #     max_length = 1.2 * nominal_length
    #     link_radius = LINK_RADIUS

    #     # 0.2 is a safety margin
    #     safety_margin = 0.12                          #for the sphere env: 0.6 + 0.2

        
    #     # loop over horizon
    #     cost_sample = 0
    #     def body(i, inputs):
    #         cost_sample, robot_states, obstaclesX = inputs

    #         # get robot state
    #         robot_state = robot_states[:, i]
    #         robot_state = robot_state.reshape(-1, NUM_OF_LINKS, 2)

    #         # Vectorize compute_end_circle
    #         compute_end_circle_batch = jax.vmap(compute_end_circle, in_axes=(0, None, None))
    #         end_centers, end_normals, _ = compute_end_circle_batch(robot_state, link_radius, nominal_length)

    #         # Compute the distance between the end centers and the goal
    #         end_center_distances = jnp.linalg.norm(goal - end_centers, axis=-1)

    #         cost_sample = cost_sample + cost_goal_coeff * end_center_distances
    #         cost_sample = cost_sample + cost_perturbation_coeff * jnp.sum((perturbed_control[:, i] - perturbation[:, i]) @ control_cov_inv * (perturbed_control[:, i] - perturbation[:, i]), axis=-1)

    #         # Vectorize state_to_config
    #         state_to_config_batch = jax.vmap(state_to_config, in_axes=(0, None, None))
    #         robot_configs = state_to_config_batch(robot_state.reshape(-1, NUM_OF_LINKS * 2), link_radius, nominal_length)

    #         # Vectorize evaluate_model
    #         evaluate_model_batch = jax.vmap(evaluate_model, in_axes=(None, 0, 0, None, None, None))
    #         csdf_distances = evaluate_model_batch(jax_params, robot_configs, robot_state.reshape(-1, NUM_OF_LINKS * 2), link_radius, nominal_length, obstaclesX)

    #         cost_sample = cost_sample + cost_safety_coeff / jnp.maximum(jnp.min(csdf_distances, axis=-1) - obst_radius, 0.01)

    #         # Compute the state constraint violation cost
    #         edge_lengths = jnp.concatenate((robot_state, 3 * nominal_length - jnp.sum(robot_state, axis=-1, keepdims=True)), axis=-1)
    #         state_constraint_violations = jnp.maximum(min_length - edge_lengths, 0) + jnp.maximum(edge_lengths - max_length, 0)
    #         state_constraint_violation = jnp.sum(state_constraint_violations)

    #         cost_sample = cost_sample + cost_state_coeff * state_constraint_violation

    #         # Update robot states
    #         robot_states = robot_states.at[:, i + 1].set(robot_dynamics_step(robot_states[:, i], perturbed_control[:, i]))

    #         return cost_sample, robot_states
        
    #     cost_sample, robot_states = lax.fori_loop(0, horizon-1, body, (cost_sample, robot_states, obstaclesX))


    #     robot_state = robot_states[:,[horizon-1]]

    #     robot_state = robot_state.reshape(NUM_OF_LINKS, 2)

    #     # Compute the end center and normal for the final state

    #     end_center, _, _ = compute_end_circle(robot_state, link_radius, nominal_length)

    #     # Compute the distance between the end center and the goal for the final state
    #     end_center_distance = jnp.linalg.norm(goal - end_center)

    #     cost_sample = cost_sample + cost_goal_coeff_final * end_center_distance

    #     cost_sample = cost_sample + cost_perturbation_coeff * ((perturbed_control[:, [horizon]]-perturbation[:,[horizon]]).T @ control_cov_inv @ perturbation[:,[horizon]])[0,0]
        
    #     robot_config = state_to_config(robot_state.squeeze(), link_radius, nominal_length)


    #     csdf_distances = evaluate_model(jax_params, robot_config, robot_state.squeeze(), link_radius, nominal_length, obstaclesX)
        
    #     cost_sample = cost_sample + cost_safety_coeff_final / jnp.max(jnp.array([jnp.min(csdf_distances) - safety_margin, 0.01]))

    #     return cost_sample, robot_states



    @jit
    def rollout_states_foresee(robot_init_state, perturbed_control, goal, obstaclesX, perturbation):

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
                cost_sample, robot_states_sample = single_sample_rollout(goal, robot_states_init, perturbed_control_sample.T, obstaclesX, perturbation_sample.T)
                return cost_sample, robot_states_sample
            batched_body_sample = jax.vmap( body_sample, in_axes=0 )
            # print('gpu used')
            cost_total, robot_states, = batched_body_sample( robot_states[:,:,0], perturbed_control, perturbation)
        else:
            @jit
            def body_samples(i, inputs):
                robot_states, cost_total, obstaclesX = inputs     

                # Get cost
                cost_sample, robot_states_sample = single_sample_rollout(goal, robot_states[i,:,0], perturbed_control[i,:,:].T, obstaclesX, perturbation[i,:,:].T)
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
    def compute_rollout_costs( key, U, init_state, goal, obstaclesX):

        perturbation, perturbed_control = compute_perturbed_control(key, control_mu, control_cov, control_bound, U)

        sampled_robot_states, costs= rollout_states_foresee(init_state, perturbed_control, goal, obstaclesX, perturbation)

        U = weighted_sum( U, perturbation, costs)

        states_final = rollout_control(init_state, U)              
        action = U[0,:].reshape(-1,1)
        U = jnp.append(U[1:,:], U[[-1],:], axis=0)

        sampled_robot_states = sampled_robot_states.reshape(( robot_n*samples, horizon ))
   
        return sampled_robot_states, states_final, action, U
    
    return compute_rollout_costs
    
def main():

    return None


if __name__=="__main__":

    main()

    
        