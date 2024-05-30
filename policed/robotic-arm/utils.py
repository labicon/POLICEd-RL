"""
Function utils for the TD3 POLICEd applied on the continuous linear environmnent.
"""

import copy, time
import torch
import numpy as np
import matplotlib.pyplot as plt
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def newFunc():
    print("Exists!")


def plot_setup(env):
    """Setups the plot environment with constraints and target."""

    ### Target
    target_circle = plt.Circle((env.target[0,0].item(), env.target[0,1].item()), env.precision, label = 'target', color='cyan', zorder=2)
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_patch(target_circle)
    ax.axis('equal')
    ax.axis('off')
    
    ### Forbidden area
    plt.plot(env.constraint_line_x1, env.constraint_line_x2, label='constraint', c='purple')
    dx = 0.05
    for x in np.arange(start=env.constraint_line_x1[0], stop=env.constraint_line_x1[1]-dx/2-1e-3, step=dx):
        plt.plot([x, x + dx], env.constraint_line_x2[0] + [0, dx], c='purple')
    # plt.plot(env.constraint_line_x1[0] + [0, dx/2], env.constraint_line_x2[0] + [dx/2, dx], c='purple')
    # plt.plot(env.constraint_line_x1[1] + [-dx/2, 0], env.constraint_line_x2[0] + [0, dx/2], c='purple')
    
    ### Constraint box
    constraint_box = torch.cat((env.allowed_constraint_area, env.allowed_constraint_area[0].unsqueeze(dim=0)), dim=0)
    plt.plot(constraint_box[:,0].numpy(), constraint_box[:,1].numpy(), label='affine region', c='red')
    
    plt.xlim([env.x_min[0,0]-0.02, env.x_max[0,0]+0.1])
    plt.ylim([env.x_min[0,1]-0.02, env.x_max[0,1]+0.1])
    
    fig.set_size_inches(4*(env.x_max[0,0]-env.x_min[0,0]), 4*(env.x_max[0,1]-env.x_min[0,1]))
    
    plt.legend(loc='lower left')


    
def plot_trajs(env, Trajectories, title=''):
    """Plots the 2D trajectories."""
    
    nb_traj = len(Trajectories) # number of trajectories to plot
    
    ### First trajectory with labels
    plt.plot(Trajectories[0][:,0].numpy(), Trajectories[0][:,1].numpy(), label='trajectory', c='blue')
    plt.scatter(Trajectories[0][0,0].item(), Trajectories[0][0,1].item(), s=20, label='start', c='green', zorder=2)
    
    ### Other trajectories without labels
    for i in range(1, nb_traj):
        if Trajectories[i].shape[0] < 1:
            continue
        plt.plot(Trajectories[i][:,0].numpy(), Trajectories[i][:,1].numpy(), c='blue')
        plt.scatter(Trajectories[i][0,0].item(), Trajectories[i][0,1].item(), s=20, c='green', zorder=2)
    
    ### Shared by all trajectories
    plt.title(title + ' Trajectory')
    plot_setup(env)
    plt.show()



def policy_tests(env, policy, N_step, initial_states, title=''):
    """Testing the policy on a trajectory propagation of length N_step.
    Plots the trajectory and displays whether constraint is respected."""
   
    nb_traj = initial_states.shape[0] # number of trajectories to propagate and plot
    Trajectories = []
    
    for i in range(nb_traj):
        state = env.reset_to_state(initial_states[i])
    
        States = torch.zeros((N_step, env.state_size))
        States[0] = state
        
        ### Trajectory propagation
        for step in range(1, N_step):
            with torch.no_grad():
                u = policy.select_action(state)
            state, reward, done, respect = env.step(u)
            States[step] = state
            if done: break
            
        if step == 0:
            print("First step is violating the constraint, nothing to plot.")
            
        print(f"Initial state: [{initial_states[i,0,0].item():.2f}, {initial_states[i,0,1].item():.2f}] \t Constraint respect: {respect} \t Final reward: {reward:.3f}")
        Trajectories.append(States[:step+1])
        
    plot_trajs(env, Trajectories, title)


    
def activation_pattern(net, x):
    """Returns the activation pattern tensor of a NN net."""
    AP = torch.zeros((net.nb_layers-1, net.hidden_size))
    
    for layer_id in range(net.nb_layers-1):
        W, b = net.layer_W_b(layer_id)
        x = x @ W.T + b
        AP[layer_id,:] = (x > 0).to(int)
        x = torch.relu(x)
    return AP

def local_affine_map(net, x):
    """Returns tensors C and d such that the NN net
    is equal to C*x + d in the activation area of x."""
    prod = torch.eye((net.layer0.in_features + 1)).to(device) # initial dim of weight + bias
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
    def __init__(self, policy):
        super().__init__()
        self.nb_layers = policy.nb_layers
        self.hidden_size = policy.hidden_size
        self.constraint_area = policy.constraint_area.clone()
        self.C = self.constraint_area.shape[0]
        self.setup(policy)
        
    def setup(self, policy):
        # Copy all layers from policy
        x = self.constraint_area
        for i in range(self.nb_layers):
            layer = getattr(policy, f"layer{i}")
            W = layer.weight
            b = layer.bias
            setattr(self, f"layer{i}", torch.nn.Linear(W.shape[1], W.shape[0]) )
            copied_layer = getattr(self, f"layer{i}")
            copied_layer.weight.data = copy.deepcopy(W)
            copied_layer.bias.data = copy.deepcopy(b)
            
            # Add the extra-bias from policy
            h = x @ W.T + b
            
            agreement = h > 0
            invalid_ones = agreement.all(0).logical_not_().logical_and_(agreement.any(0))
            sign = agreement[:, invalid_ones].half().sum(0).sub_(self.C / 2 + 0.01).sign_()
            extra_bias = (h[:, invalid_ones] * sign - 1e-4).amin(0).clamp(max=0) * sign
            copied_layer.bias.data[invalid_ones] -= extra_bias
            h[:, invalid_ones] -= extra_bias
            x = torch.relu(h)
        
    def forward(self, x):
        for i in range(self.nb_layers-1):
            x = getattr(self, f"layer{i}")(x)
            x = torch.relu(x)
        x = getattr(self, f"layer{self.nb_layers-1}")(x)
        return x
    
    def layer_W_b(self, layer_id):
        """Returns weight and bias of given layer."""
        layer = getattr(self, f"layer{layer_id}")
        return layer.weight.data, layer.bias.data
    
    def select_action(self, state):
        return (self.forward(state.to(device))).cpu()
    
    

def activation_map(policy, env, delta_x = 0.1):
    """Plots the different activation patters with different colors
    over the constraint_area.
    Returns the number of switches of activation patterns."""
    
    d = ((env.x_max - env.x_min)/delta_x).squeeze().to(int)
    N_x = d[0].item()
    N_y = d[1].item()
    
    colors = np.zeros((N_x+1, N_y+1, 1, 3))
    colors[0,0] = np.array([[0., 0., 1.]])
    
    AP_map = torch.zeros((N_x+1, N_y+1, env.state_size, policy.hidden_size))
    AP_map[0,0] = activation_pattern(policy, env.x_min)
    
    nb_switches = 0 # number of activation pattern switch
    
    plt.title("Activation pattern map")
    plot_setup(env)
    plt.legend()
    
    plt.scatter(env.x_min[0,0].item(), env.x_min[0,1].item(), s=20, c=colors[0,0], zorder=3)
    
    for j in range(N_y+1):
        y = env.x_min[0,1].item() + j*delta_x
        for i in range(N_x+1):
            if i == 0 and j == 0:
                continue # first point already initialized
            x = env.x_min[0,0].item() + i*delta_x
            state = torch.tensor([[x,y]])
            AP_map[i,j] = activation_pattern(policy, state)
            
            ### Check whether we have seen this activation pattern before
            i_same_AP = i-1; j_same_AP = j # indices of point with same AP
            if i_same_AP == -1:
                j_same_AP -= 1
                i_same_AP = N_x
                
            while not (AP_map[i,j] == AP_map[i_same_AP, j_same_AP]).all(): # different activation
                i_same_AP -= 1
                if i_same_AP == -1:
                    if j_same_AP == 0: # no other stored AP is the same
                        break
                    else:
                        j_same_AP -= 1
                        i_same_AP = N_x
            
            ### Decide the color of AP
            if i_same_AP == -1 and j_same_AP == 0: # no other stored AP is the same
                nb_switches += 1
                # print(f"Switch at ({x:5.2f}, {y:5.2f})")
                colors[i,j] = np.array([[i/N_x, j/N_y, 1-i/N_x]])
            else: # same activation
                colors[i,j] = colors[i_same_AP, j_same_AP]
            ### Plot the color
            plt.scatter(x, y, s=20, c=colors[i,j], zorder=3)
            
    plt.show()
    return nb_switches



def same_activation(policy):
    """Verifies whether the whole CONVEX constraint area has the same activation pattern.
    Returns a boolean and a vertex where activation differs."""
    
    nb_vertices = policy.constraint_area.shape[0] # number of vertices of the constraint area
    ### Verify that all vertices of the constraint area have the same activation pattern
    prev_AP = activation_pattern(policy, policy.constraint_area[0])
    for vertex_id in range(1,nb_vertices):
        vertex = policy.constraint_area[vertex_id].unsqueeze(dim=0)
        AP = activation_pattern(policy, vertex)
        if not (AP == prev_AP).all(): 
            print("The constraint region does not share the same activation pattern.")
            return False, vertex
        
    print("The constraint region shares the same activation pattern.")
    return True, vertex
     

def constraint_verification(env, policy):
    """Verifies whether the constraint holds on the whole CONVEX constraint area.
    Returns a boolean, a vertex where constraint fails, and its next state violating constraint."""
    
    nb_vertices = env.allowed_constraint_area.shape[0] # number of vertices of the constraint area
    
    ### Verify that the constraint holds at every vertex of the CONVEX constraint area
    for vertex_id in range(nb_vertices):
        vertex = env.allowed_constraint_area[vertex_id].unsqueeze(dim=0)
        vertex += 1e-3 * env.width_affine_area
        env.reset_to_state(vertex)
        with torch.no_grad():
            action = policy.select_action(vertex)
        violating_state, _, _, respect = env.step(action)
        if not respect:
            print(f"Constraint is not satisfied on vertex [{vertex[0,0].item():.2f}, {vertex[0,1].item():.2f}] and violating_state [{violating_state[0,0].item():.4f}, {violating_state[0,1].item():.4f}]")
            return False, vertex, violating_state
    
    print("Constraint is satisfied everywhere in the allowed constraint area if it is convex.")
    return True, None, None


def empirical_constraint_violation(env, policy, nb_intermediaries = 1000, display = True):
    """Empirical constraint verification on all vertices and at intermediary points inbetween."""
    
    nb_violations = 0
    ### Verify that the constraint holds along the constraint_line
    v1 = env.constraint_min
    v2 = env.constraint_max
    for k in range(nb_intermediaries):
        state = v1 + (v2 - v1)*k/nb_intermediaries
        state += 1e-3 * env.width_affine_area
        env.reset_to_state(state)
        with torch.no_grad():
            action = policy.select_action(state)
        _, _, _, respect = env.step(action)
        if not respect:
            nb_violations += 1
    if display:
        print(f"Constraint is violated on {100*nb_violations/nb_intermediaries:.1f}% of the constraint line.")
    return nb_violations/nb_intermediaries

   
def Q_value_map(env, critic, title = ''):
    """Plots the Q-value arrow map showing what are the best directions
    recommended by the critic network."""
    
    eps = 1e-4
    scaling = 0.5 # arrow length scaling
    action_range = torch.arange(start=-1., step=0.05, end=1.).unsqueeze(dim=1)
    N = action_range.shape[0]
    Actions = torch.zeros((N*N, 2))
    for i in range(N):
        Actions[N*i:N*(i+1)] = torch.cat((action_range[i]*torch.ones((N,1)), action_range), dim=1)
    
    plt.title(title + ' Q-value best direction map')
    for x in np.arange(start=env.x_min[0,0].item(), stop=env.x_max[0,0].item() + eps, step=0.1):
        for y in np.arange(start=env.x_min[0,1].item(), stop=env.x_max[0,1].item() + eps, step=0.1):
            state = torch.tensor([[x, y]]).float()
            s = (torch.ones((N*N,1)) @ state).to(device)
            with torch.no_grad():
                Q_val1, Q_val2 = critic(s, Actions.to(device))
            max_id = torch.argmax(torch.min(Q_val1, Q_val2)).item()
            env.reset_to_state(state)
            next_state, _, _, _ = env.step(Actions[max_id].unsqueeze(dim=0))
            dx = scaling*(next_state - state)[0,0].item()
            dy = scaling*(next_state - state)[0,1].item()
            plt.arrow(x, y, dx, dy, width = 0.01)
    
    plot_setup(env)
    plt.show()




def policy_map(env, policy, title = ''):
    """Plots the policy arrow map showing the directions chosen by the policy."""
    
    eps = 1e-4
    scaling = 0.5 # arrow length scaling
    plt.title(title + ' Policy best direction map')
    for x in np.arange(start=env.x_min[0,0].item(), stop=env.x_max[0,0].item() + eps, step=0.1):
        for y in np.arange(start=env.x_min[0,1].item(), stop=env.x_max[0,1].item() + eps, step=0.1):
            state = torch.tensor([[x, y]]).float()
            env.reset_to_state(state)
            with torch.no_grad():
                action = policy.select_action(state)
            
            next_state, _, _, _ = env.step(action)
            dx = scaling*(next_state - state)[0,0].item()
            dy = scaling*(next_state - state)[0,1].item()
            plt.arrow(x, y, dx, dy, width = 0.01)
    
    plot_setup(env)
    plt.show()









class ReplayBuffer(object):
    """Buffer to save experiences as (state, action, next_state, reward, not_done)"""
    def __init__(self, state_dim, action_dim, max_size=int(1e6), priority_method="index", alpha=1):
        self.max_size = max_size
        self.ptr = 0 # id where next experience can be stored
        self.size = 0 # size of the buffer

        self.state = torch.zeros((max_size, state_dim))
        self.action = torch.zeros((max_size, action_dim))
        self.next_state = torch.zeros((max_size, state_dim))
        self.reward = torch.zeros((max_size, 1))
        self.not_done = torch.zeros((max_size, 1))
        self.priority_method = priority_method
        self.alpha = alpha

        print("device: ", device)


    def add(self, state, action, next_state, reward, done):
        """Adds an experience to the buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        if(self.size == 0): # Quick Sanity Check!
            print("This buffer is empty!")
            return
        
        if(False):
            start = time.time()
            self.index_priority_sample(batch_size)
            end = time.time()
            self.reward_priority_sample(batch_size)
            end2 = time.time()
            print("Index Time", (end-start))
            print("Reward Time", (end2-end))

        
        # Redirect based on priority method, or just returns a uniformly sampled 
        if(self.priority_method == "index"):
            return self.index_priority_sample(batch_size)
        elif(self.priority_method == "reward"):
            return self.reward_priority_sample(batch_size)
        else:
            ind = np.random.randint(0, self.size, size=batch_size)
            return (self.state[ind].to(device), self.action[ind].to(device), self.next_state[ind].to(device), self.reward[ind].to(device), self.not_done[ind].to(device))

    def index_priority_sample(self, batch_size):
        possible_indices = np.arange(self.size)
        priorities = np.power(possible_indices, self.alpha)
        ind = np.random.choice(possible_indices, p=np.divide(priorities, priorities.sum()), size=batch_size)
        return (self.state[ind].to(device), self.action[ind].to(device), self.next_state[ind].to(device), self.reward[ind].to(device), self.not_done[ind].to(device))
    
    def reward_priority_sample(self, batch_size):
        possible_indices = torch.arange(self.size)
        priorities = torch.pow(self.reward[:self.size]+abs(min(self.reward[:self.size]))+1.0, self.alpha).double().flatten()
        ind = np.random.choice(possible_indices, p=torch.div(priorities, priorities.sum()), size=batch_size)
        return (self.state[ind].to(device), self.action[ind].to(device), self.next_state[ind].to(device), self.reward[ind].to(device), self.not_done[ind].to(device))



    
    def remove_constraint_violations(self, env): 
        """Remove all the rows of the ReplayBuffer where the state or the next_state are
        violating the constraint."""
        
        to_clean = env.in_constraint_area(self.state)*(~env.respect_constraint(self.state)) + env.in_constraint_area(self.next_state)*(~env.respect_constraint(self.next_state))
        rows_to_remove = list(torch.where(to_clean)[0])
        nb_violations = len(rows_to_remove)
        rows_to_remove.append(self.size-1)
        
        states = torch.clone(self.state)
        actions = torch.clone(self.action)
        next_states = torch.clone(self.next_state)
        rewards = torch.clone(self.reward)
        not_done = torch.clone(self.not_done)
        
        ### Erase violating rows by moving back all following rows
        for i in range(nb_violations):
            start_row = rows_to_remove[i]
            end_row = rows_to_remove[i+1]
            self.state[start_row-i:end_row-i] = states[start_row+1:end_row+1]
            self.action[start_row-i:end_row-i] = actions[start_row+1:end_row+1]
            self.next_state[start_row-i:end_row-i] = next_states[start_row+1:end_row+1]
            self.reward[start_row-i:end_row-i] = rewards[start_row+1:end_row+1]
            self.not_done[start_row-i:end_row-i] = not_done[start_row+1:end_row+1]
        
        if nb_violations >= 1:
            self.state[end_row-i:end_row+1] = torch.zeros_like(self.state[end_row-i:end_row+1])
            self.action[end_row-i:end_row+1] = torch.zeros_like(self.action[end_row-i:end_row+1])
            self.next_state[end_row-i:end_row+1] = torch.zeros_like(self.next_state[end_row-i:end_row+1])
            self.reward[end_row-i:end_row+1] = torch.zeros_like(self.reward[end_row-i:end_row+1])
            self.not_done[end_row-i:end_row+1] = torch.zeros_like(self.not_done[end_row-i:end_row+1])
        
        ### Update size and current experience index
        self.ptr -= nb_violations
        self.size -= nb_violations
        
        
        
from collections import deque       


class episode_stats_queue(object):
    """Queue to store the rewards of the last 100 episodes."""
    def __init__(self, max_size = int(100)):
        self.reward_queue = deque()
        self.respect_queue = deque()
        self.completion_queue = deque()
        self.size = 0
        self.max_size = max_size

    def add(self, episode_reward, episode_respect, episode_completion):
        """Adds an element to the end of the queue."""
        self.reward_queue.append(episode_reward)
        self.respect_queue.append(episode_respect)
        self.completion_queue.append(episode_completion)
        if self.size == self.max_size: # max size, pop first one, size doesn't change
            self.reward_queue.popleft()
            self.respect_queue.popleft()
            self.completion_queue.popleft()
        else: # not max size, increase it
            self.size += 1
            
    def average(self):
        """Calculates the average episode reward and average episode respect."""
        return sum(self.reward_queue)/self.size, sum(self.respect_queue)/self.size
            
    def percentCompletion(self):
        return 100.0*sum(self.completion_queue)/self.size























     