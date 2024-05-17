# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:52:25 2023

@author: Jean-Baptiste Bouvier

Function utils for the PPO POLICEd applied on the Inverted Pendulum environmnent.
"""

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#%% Utils to verify that the POLICEd network is affine in the buffer region

def activation_pattern(net, x):
    """Returns the activation pattern tensor of a NN net."""
    AP = torch.zeros((net.nb_layers-1, net.width))
    
    for layer_id in range(net.nb_layers-1):
        W, b = net.layer_W_b(layer_id)
        x = x @ W.T + b
        AP[layer_id,:] = (x > 0).to(int)
        x = torch.relu(x)
    return AP

def local_affine_map(net, x):
    """Returns tensors C and d such that the NN net
    is equal to C*x + d in the activation area of x."""
    prod = torch.eye((net.layer_0.in_features + 1)).to(device) # initial dim of weight + bias
    # weight + bias activation
    weight_AP = activation_pattern(net, x)
    bias_AP = torch.ones((net.nb_layers-1,1))
    AP = torch.cat((weight_AP, bias_AP), dim=1).to(device)
    
    for layer_id in range(net.nb_layers-1):
        W, b = net.layer_W_b(layer_id)
        row = torch.cat((torch.zeros((1,W.shape[1])), torch.tensor([[1.]])), dim=1).to(device)
        W_tilde = torch.cat( (torch.cat((W, b.unsqueeze(dim=1)), dim=1), row), dim=0)
        prod = torch.diag_embed(AP[layer_id]) @ W_tilde @ prod
        
    W, b = net.layer_W_b(net.nb_layers-1)
    W_tilde = torch.cat((W, b.unsqueeze(dim=1)), dim=1)
    prod = W_tilde @ prod
    return prod[:,:-1], prod[:,-1]


class NetworkCopy(torch.nn.Module):
    """Creates a copy of the ConstrainedPolicyNetwork but without the extra-bias
    computation, which is simply included in the layer biases."""
    def __init__(self, policy, state_norm):
        super().__init__()
        self.nb_layers = policy.nb_layers
        self.width = policy.width
        self.state_norm = state_norm
        self.buffer_vertices = self.state_norm(policy.buffer_vertices.clone(), update=False)
        self.num_vertices = self.buffer_vertices.shape[0]
        self.setup(policy)
        
    def setup(self, policy):
        # Copy all layers from policy
        x = self.buffer_vertices
        for i in range(self.nb_layers):
            layer = getattr(policy, f"layer_{i}")
            W = layer.weight
            b = layer.bias
            setattr(self, f"layer_{i}", torch.nn.Linear(W.shape[1], W.shape[0]) )
            copied_layer = getattr(self, f"layer_{i}")
            copied_layer.weight.data = copy.deepcopy(W)
            copied_layer.bias.data = copy.deepcopy(b)
            
            # Add the extra-bias from policy
            h = x @ W.T + b
            
            agreement = h > 0
            invalid_ones = agreement.all(0).logical_not_().logical_and_(agreement.any(0))
            sign = agreement[:, invalid_ones].float().sum(0).sub_(self.num_vertices / 2 + 1e-3).sign_()
            extra_bias = (h[:, invalid_ones] * sign - 1e-4).amin(0).clamp(max=0) * sign
            copied_layer.bias.data[invalid_ones] -= extra_bias
            h[:, invalid_ones] -= extra_bias
            x = torch.relu(h)
        
    def forward(self, x):
        x = self.state_norm(x, update=False)
        for i in range(self.nb_layers-1):
            x = getattr(self, f"layer_{i}")(x)
            x = torch.relu(x)
        x = getattr(self, f"layer_{self.nb_layers-1}")(x)
        return x
    
    def layer_W_b(self, layer_id):
        """Returns weight and bias of given layer."""
        layer = getattr(self, f"layer_{layer_id}")
        return layer.weight.data, layer.bias.data
    


def same_activation(policy):
    """Verifies whether the whole CONVEX constraint area has the same activation pattern.
    Returns a boolean and a vertex where activation differs."""
    
    nb_vertices = policy.buffer_vertices.shape[0] # number of vertices of the constraint area
    ### Verify that all vertices of the constraint area have the same activation pattern
    prev_AP = activation_pattern(policy, policy.buffer_vertices[0])
    for vertex_id in range(1,nb_vertices):
        vertex = policy.buffer_vertices[vertex_id].unsqueeze(dim=0)
        # Vertices in copied network already normalized
        AP = activation_pattern(policy, vertex)
        if not (AP == prev_AP).all(): 
            print("The constraint region does not share the same activation pattern.")
            return False, vertex
        
    print("The constraint region shares the same activation pattern.")
    return True, vertex
     





#%% Utils for POLICEd RL theory: calculate minimal buffer radius and linearization constant

def buffer_vertices(args, eval_env, nb_steps=10, display=False):
    """First calculates the minimal buffer radius such that
    the buffer cannot be jumped over in one timestep.
    Returns the corresponding buffer vertices and the radius.
    nb_steps is the number of steps for each state in the grid search for the min radius"""

    eps = 1e-10
    ### Matrices for the constraint
    C = args.constraint_C
    assert C[0]**2 + C[2]**2 < eps, "buffer radius search needs C = [0 b 0 1]"
    assert abs(C[3] - 1.) < eps, "buffer radius search needs C = [0 b 0 1]"
    b = C[1]
    assert b > 0., "buffer radius search needs C[1] > 0"
    assert args.constraint_d < eps, "buffer radius search needs d = 0"
    
    ### Unpack buffer bounds for the non-constrained states (i.e. except theta_dot)
    x_min, theta_min, x_dot_min, _ = args.min_state
    x_max, theta_max, x_dot_max, _ = args.max_state
    
    r = 0.1 # initial guess for the buffer radius
    step = (args.max_state - args.min_state)/nb_steps
    x_range     = np.arange(start = x_min,     stop = x_max     + step[0]/2, step = step[0])
    theta_range = np.arange(start = theta_min, stop = theta_max + step[1]/2, step = step[1])
    x_dot_range = np.arange(start = x_dot_min, stop = x_dot_max + step[2]/2, step = step[2])
    action_range = np.array([[-eval_env.action_max], [eval_env.action_max]])
    
    if display:
        print("Iterative search of the minimal buffer radius r")
    for repeats in range(4):   
        max_r = 0.
        for x in x_range:
            for theta in theta_range:
                theta_dot = -b*theta -r
                for x_dot in x_dot_range:
                    state = np.array([x, theta, x_dot, theta_dot])
                    for a in action_range:
                        eval_env.reset_to(state)
                        next_state, _, _, _ = eval_env.step(a)
                        dif = C @ (next_state - state)
                        if dif > max_r:
                            max_r = dif
        if display: # shows convergence of the buffer radius
            print(f"r = {r:.4f} \t max_r = {max_r:.4f}")
        r = max_r
        
    ### Buffer vertices calculation
    v = np.zeros((16, 4))
    i = 0
    for x in [x_min, x_max]:
        for theta in [theta_min, theta_max]:
            for x_dot in [x_dot_min, x_dot_max]:
                for theta_dot in [-r-b*theta, -b*theta]:
                    v[i] = np.array([x, theta, x_dot, theta_dot])
                    i += 1        
    return v, r




def epsilon(args, env):
    """
    Calculates the epsilon difference for the dynamics linearization.
    A lot of specifics to the pendulum dynamics and constraint.
    """
    
    C = args.constraint_C
    N = int(2e3) # number of test points
    Vertices = args.buffer_vertices
    nb_vertices = Vertices.shape[0]
    
    assert C[0]**2 + C[2]**2 < 1e-8, "buffer radius search needs C = [0 b 0 1]"
    assert abs(C[3] - 1.) < 1e-8, "buffer radius search needs C = [0 b 0 1]"
    b = C[1]
    assert b > 0., "buffer radius search needs C[1] > 0"
    
    ### Storing datapoints to perform linear regression on
    States = np.zeros((N, env.state_size))
    Next_States = np.zeros((N, env.state_size))
    Actions = np.zeros((N, env.action_size))
    
    for i in range(N):
        ### Random coefficients for a weighted sum of vertices
        if i % 2 == 0:
            coefs = np.random.rand(nb_vertices)
            coefs /= coefs.sum() # always all around 0
        else:
        # coef of decreasing size, then shuffled. Allows larger coefficients
            coefs = np.random.rand(nb_vertices)
            for j in range(1, nb_vertices):
                cum_sum = coefs[:j].sum()
                coefs[j] *= 1 - cum_sum
            np.random.shuffle(coefs)
            
        ### Random state and actions
        States[i] = coefs @ Vertices
        Actions[i] = np.random.rand(env.action_size)*2*env.action_max - env.action_max
        ### Propagation
        env.reset_to(States[i])
        Next_States[i], _, _, _ = env.step(Actions[i])

    
    ############## Uses assumptions on dynamics
    ### Linear regression
    Ones = np.ones((N, 1))
    A = np.concatenate((States, Actions, b*Ones, Ones), axis=1)
    Cs_dot = (Next_States - States)/env.dt @ C.T
    B = Cs_dot - b*States[:, -1]
    x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    ### Calculation of epsilon
    CA = np.array([[x[0], x[1], x[2], x[3]+b]])
    CB = x[4]
    Cc = b*x[5] + x[6]
    affine_pred = States @ CA.T + Actions * CB.T + Cc
    eps1 = abs(Cs_dot - affine_pred.squeeze()).max()
    
    ################# No assumptions on the dynamics
    Ones = np.ones((N, 1))
    A = np.concatenate((b*States, States, b*Actions, Actions, b*Ones, Ones), axis=1)
    Cs_dot = (Next_States - States)/env.dt @ C.T
    x, _, _, _ = np.linalg.lstsq(A, Cs_dot, rcond=None)
    
    ### Calculation of epsilon
    CA = np.array([[b*x[0] + x[4], b*x[1] + x[5], b*x[2] + x[6], b*x[3] + x[7]]])
    CB = b*x[8] + x[9]
    Cc = b*x[10] + x[11]
    affine_pred = States @ CA.T + Actions * CB.T + Cc
    eps2 = abs(Cs_dot - affine_pred.squeeze()).max()
    
    ### both epsilons are very close
    return max(eps1, eps2)





#%% Training utils

class ReplayBuffer:
    """Store experiences. The sampling from this buffer is handled by the update function
    of PPO, where each experience is sampled as part of mini-batches."""
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))
        self.a = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


def evaluate_policy(args, env, agent, state_norm):
    """Evaluates the policy, returns the average reward and whether the
    reward threshold has been met"""
    times = 1#3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.evaluate(s, state_norm)  # We use the deterministic policy during the evaluating
            s, r, done, _ = env.step(action)
            episode_reward += r
        evaluate_reward += episode_reward

    average_reward = evaluate_reward / times
    return average_reward, average_reward >= env.reward_threshold


def plot_traj(env, Traj, N_step=None):
    """Plots the trajectory of the states of the inverted pendulum."""
    if N_step is None:
        N_step = env.max_episode_steps
        
    N = Traj.shape[0] # number of steps of the trajectory before terminating
    
    titles = ["Cartpole position (m)", "Pole angle (rad)", "Cartpole speed (m/s)", "Pole speed (rad/s)"]
    variables = ["x", "theta", "x dot", "theta dot"]

    for state_id in range(Traj.shape[1]):
        
        plt.title(titles[state_id])
        plt.scatter(np.arange(N), Traj[:, state_id], s=10, label=variables[state_id])
        
        if env.max_state[state_id] < 1e5: # don't display infinite limit
            plt.plot(np.array([0, N_step]),  np.ones(2)*env.max_state[state_id], c='red')
            plt.plot(np.array([0, N_step]), -np.ones(2)*env.max_state[state_id], c='red', label="limit")
        
        plt.legend()
        plt.show()


 
def training(args, env, agent, replay_buffer, env_eval, state_norm=None, reward_scaling=None, data=None):
    """Training of the PPO agent to achieve high reward"""
    
    evaluate_num = 0  # Record the number of evaluations
    if hasattr(args, 'total_steps'):
        total_steps = args.total_steps
    else:
        total_steps = 0  # Record the total steps during the training
    stable = False # Inverse pendulum not stabilized
    trained = False # Stable and buffer reached full size
    episode = 0
    buffer_vertices = args.buffer_vertices
    num_vertices = buffer_vertices.shape[0]
    
    while (not trained):
        episode += 1
        
        if episode % 5 == 0: # reset in buffer regularly            
            coefs = np.random.rand(num_vertices)
            coefs /= coefs.sum()
            s = env.reset_to(coefs @ buffer_vertices)
        else:
            s = env.reset()
        
        if args.use_reward_scaling:
            reward_scaling.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s, state_norm)  # Action and the corresponding log probability
            s_, r, done, _ = env.step(a)

            if args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be True, we need to distinguish them;
            # dw means dead or win, there is no next state s'
            # but when reaching the max_episode_steps, there is a next state s' actually.
            if done and episode_steps != env.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size, then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps, state_norm)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, stable = evaluate_policy(args, env_eval, agent, state_norm)
                print(f"evaluation {evaluate_num} \t reward: {evaluate_reward:.1f}")
                data.plot()
            
            if stable and args.POLICEd and args.enlarging_buffer and agent.actor.iter == 0:
                print("\nStarting to enlarge the buffer")
                agent.actor.enlarge_buffer = True
            
            trained = total_steps > args.max_train_steps or stable
            if args.POLICEd and args.enlarging_buffer:
                trained = trained and agent.actor.iter > agent.actor.max_iter
           
        r = respect_ratio(args, env_eval, agent, state_norm)
        data.add(episode_steps, r)
        
    args.total_steps = total_steps
    return stable




#%% Utils to verify and train for constraint respect

def constraint_verification(args, env, agent, state_norm, display=True):
    """Verifies whether the constraint hold by checking each vertices of the buffer.
    Returns a Boolean of respect and if violation returns the id of the violating vertex"""
    
    tol = args.tol # = -2*eps*dt
    C = args.constraint_C
    nb_vertices = args.buffer_vertices.shape[0] # number of vertices of the constraint area
    
    ### Verify that the constraint holds at every vertex of the CONVEX constraint area
    for vertex_id in range(nb_vertices):
        vertex = args.buffer_vertices[vertex_id]
        env.reset_to(vertex)
            
        with torch.no_grad():
            action = agent.evaluate(vertex, state_norm)
        next_state, _, _, _ = env.step(action)
        if C @ (next_state - vertex) > tol:
            if display:
                print(f"Constraint is not satisfied on vertex {vertex_id}")
            return False, vertex_id
    if display:
        print("Constraint is satisfied everywhere")
    return True, vertex_id




def constraint_training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling=None, data=None):
    """Constraint training of the PPO agent."""
    
    respect = False
    buffer_vertices = args.buffer_vertices
    num_vertices = buffer_vertices.shape[0]
    C = args.constraint_C
    id_bad_vertex = 0
    if hasattr(args, 'total_steps'):
        total_steps = args.total_steps
    else:
        total_steps = 0
    repeats = 0
    while not respect and repeats < 4:
        repeats += 1
        
        if args.use_reward_scaling:
            reward_scaling.reset()
        
        ### Reset on each vertex of the buffer and take one step
        for vertex_id in range(id_bad_vertex, num_vertices):
            total_steps += 1
            s = env.reset_to(buffer_vertices[vertex_id])
            
            a, a_logprob = agent.choose_action(s, state_norm)  # Action and the corresponding log probability
            s_, r, done, vertex_respect = env.step(a)
            value = C @ (s_ - s)
            
            if value > args.tol: # vertices where constraint is not respected
                r = 5*(args.tol - value) -1.
                if args.use_reward_scaling:
                    r = reward_scaling(r)
                # print(f"Vertex {vertex_id},\t v = {value:.3f} > tol = {args.tol:.3f}")
                dw = done # Since there is only one step died/win is equal to done
                replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
                if replay_buffer.count == args.batch_size:
                    replay_buffer.count = 0
        
        if replay_buffer.count > 1: # don't update with empty buffer
            agent.update(replay_buffer, total_steps, state_norm)
        respect, id_bad_vertex = constraint_verification(args, env_eval, agent, state_norm, display=True)
    
    evaluate_reward, stable = evaluate_policy(args, env_eval, agent, state_norm)
    r = respect_ratio(args, env_eval, agent, state_norm)
    data.add(evaluate_reward, r)
    print(f"Reward after constraint training: {evaluate_reward:.2f}")
    args.total_steps = total_steps
    return stable, respect




def respect_ratio(args, env, agent, state_norm):
    """Empirical constraint verification
    Returns a percentage of constraint respect over a griding of the buffer."""
    tol = args.tol # = -2*eps*dt
    C = args.constraint_C
    nb_violations = 0
    
    nb_steps = 3
    step = (args.max_state - args.min_state)/nb_steps
    x_min, theta_min, x_dot_min, theta_dot_min = args.min_state
    x_max, theta_max, x_dot_max, theta_dot_max = args.max_state
    x_range         = np.arange(start = x_min,         stop = x_max         + step[0]/2, step = step[0])
    theta_range     = np.arange(start = theta_min,     stop = theta_max     + step[1]/2, step = step[1])
    x_dot_range     = np.arange(start = x_dot_min,     stop = x_dot_max     + step[2]/2, step = step[2])
    theta_dot_range = np.arange(start = theta_dot_min, stop = theta_dot_max + step[3]/2, step = step[3])
    nb_points = len(x_range)*len(theta_range)*len(x_dot_range)*len(theta_dot_range)
    
    for x in x_range:
        for theta in theta_range:
            for x_dot in x_dot_range:
                for theta_dot in theta_dot_range:
                    state = np.array([x, theta, x_dot, theta_dot])
                    env.reset_to(state)
                        
                    with torch.no_grad():
                        action = agent.evaluate(state, state_norm)
                    next_state, _, _, _ = env.step(action)
                    if C @ (next_state - state) > tol:
                        nb_violations += 1
           
    return 1 - (nb_violations/nb_points)
   
class data:
    """class to store the rewards over time and the percentage of constraint respect
    during training to plot it"""
    def __init__(self):
        self.len = 0
        self.reward_list = []
        self.respect_list = []
        
    def add(self, reward, respect):
        self.len += 1
        self.reward_list.append(reward)
        self.respect_list.append(respect)
        
    def add_respect(self, respect):
        self.respect_list.append(respect)
        
    def plot(self):
        iterations = np.arange(self.len)
        plt.title("Rewards")
        plt.plot(iterations, np.array(self.reward_list))
        plt.xlabel("Episodes")
        plt.show()
        
        iterations = np.arange(len(self.respect_list))
        plt.title("Respect during training")
        plt.plot(iterations, np.array(self.respect_list))
        plt.xlabel("Episodes")
        plt.show()
        
        








def propagate(args, env, agent, state_norm, initial_state=None, traj_len=None):
    """Propagate a trajectory from the initial state with the agent."""
    
    if initial_state is None:
        state = env.reset()
    else:
        state = env.reset_to(initial_state)
     
    if traj_len is None:
        traj_len = env.max_episode_steps
     
    episode_reward = 0
    Trajectory = np.zeros((env.max_episode_steps, env.state_size))
    Trajectory[0] = state
    
    for t in range(1, traj_len):
        with torch.no_grad():
            action = agent.evaluate(state, state_norm)
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        Trajectory[t] = state
        if done: break
    
    return Trajectory[:t+1], episode_reward
