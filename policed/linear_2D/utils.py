"""
Created on Fri Nov 27 10:09:16 2023

@author: Jean-Baptiste Bouvier

Function utils for the TD3 POLICEd applied on the continuous linear environmnent.
"""

import copy
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





#%% Utils to verify the affine character of the trained POLICEd policy
    
def activation_pattern(net, x):
    """Returns the activation pattern tensor of a NN net for an input x."""
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
    prod = torch.eye((net.layer0.in_features + 1)) # initial dim of weight + bias
    # weight + bias activation
    weight_AP = activation_pattern(net, x)
    bias_AP = torch.ones((net.nb_layers-1,1))
    AP = torch.cat((weight_AP, bias_AP), dim=1)
    
    for layer_id in range(net.nb_layers-1):
        W, b = net.layer_W_b(layer_id)
        row = torch.cat((torch.zeros((1,W.shape[1])), torch.tensor([[1.]])), dim=1)
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
    
    

def activation_map(policy, env):
    """Plots the different activation patters with different colors
    over the constraint_area.
    Returns the number of switches of activation patterns."""
    
    delta_x = 0.01
    d = ((env.x_max - env.x_min)/delta_x).squeeze().to(int)
    N_x = d[0].item()
    N_y = d[1].item()
    
    colors = np.zeros((N_x+1, N_y+1, 1, 3))
    colors[0,0] = np.array([[0., 0., 1.]])
    
    AP_map = torch.zeros((N_x+1, N_y+1, env.state_size, policy.hidden_size))
    AP_map[0,0] = activation_pattern(policy, env.x_min)
    
    nb_switches = 0 # number of activation pattern switch
    
    plt.title("Activation pattern map")
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.axis('equal')
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')   
    ax.set_yticks([])
    ax.set_xticks([])
    plt.tight_layout()
    fig.set_size_inches(4, 4)
    
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
   

    
#%% Utils to verify whether the policy respects the constraint

def constraint_verification(env, policy):
    """Verifies whether the constraint holds on the whole CONVEX constraint area.
    Returns a boolean, a vertex where constraint fails, and its next state violating constraint."""
    
    nb_vertices = env.allowed_constraint_area.shape[0] # number of vertices of the constraint area
    
    ### Verify that the constraint holds at every vertex of the CONVEX constraint area
    for vertex_id in range(nb_vertices):
        vertex = env.allowed_constraint_area[vertex_id].unsqueeze(dim=0)
        vertex += 1e-3 * env.width_affine_area
        env.reset_to(vertex)
        with torch.no_grad():
            action = policy.select_action(vertex)
        violating_state, _, _, respect = env.step(action)
        if not respect:
            print(f"Constraint is not satisfied on vertex [{vertex[0,0].item():.2f}, {vertex[0,1].item():.2f}] and violating_state [{violating_state[0,0].item():.4f}, {violating_state[0,1].item():.4f}]")
            return False, vertex, violating_state
    
    print("Constraint is satisfied everywhere in the allowed constraint area if it is convex.")
    return True, None, None


def empirical_constraint_violation(env, policy, nb_test_points=1000, display=False):
    """Empirical constraint verification on all vertices and at intermediary points inbetween."""
    
    nb_violations = 0
    ### Verify that the constraint holds along the constraint_line
    v1 = env.constraint_min
    v2 = env.constraint_max
    for k in range(nb_test_points):
        state = v1 + (v2 - v1)*k/nb_test_points
        state += 1e-3 * env.width_affine_area
        env.reset_to(state)
        with torch.no_grad():
            action = policy.select_action(state)
        next_state, _, _, respect = env.step(action)
        if not respect:
            nb_violations += 1
    if display:
        print(f"Constraint is violated on {100*nb_violations/nb_test_points:.1f}% of the constraint line.")
    return nb_violations/nb_test_points

   



#%% Various plot utils


def background(env, POLICEd):
    """Plots the background of the policy map"""
    
    fig = plt.gcf()
    target_color = 'cyan'
    constraint_color = 'red'
    buffer_color = (144./255, 238./255, 144./255)

    ax = fig.gca()
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']

    if POLICEd:
        buffer = plt.Rectangle(xy=[0.36, 0.6], width=0.66, height=0.11, color=buffer_color)
        ax.add_patch(buffer)
        buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='s', linestyle='None', markersize=10, label='affine buffer $\mathcal{B}$')
        
    ### Target
    target_circle = plt.Circle((env.target[0,0].item(), env.target[0,1].item()), env.precision,
                               color=target_color, zorder=3)
    ax.add_patch(target_circle)
    ax.axis('equal')
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')

    ### Forbidden area
    dy = 0.015
    plt.plot(np.array([0.4-0.03, 1+0.015]), env.constraint_line_x2 + dy, linewidth='4', c=constraint_color)
    dx = 0.05
    for x in np.arange(start=env.constraint_line_x1[0]-0.03, stop=env.constraint_line_x1[1]-dx/2-1e-3+0.01, step=dx):
        plt.plot([x, x + dx], env.constraint_line_x2[0] + [0, dx] + dy, linewidth='2', c=constraint_color)

    target_marker = mlines.Line2D([], [], color=target_color, marker='o', linestyle='None', markersize=10, label='target')
    constraint_marker = mlines.Line2D([], [], color=constraint_color, linewidth='4', markersize=10, label='constraint line')
    legend_handle = [target_marker, constraint_marker]
    if POLICEd:
        legend_handle.append(buffer_marker)

    plt.legend(handles = legend_handle,
               loc='lower left', frameon=False, borderpad=.0,
               labelspacing=0.3, handletextpad=0.5, handlelength=1.4)

    ax.set_yticks([])
    ax.set_xticks([])
    plt.xlim([-0.02, 1.055])
    plt.ylim([-0.02, 1.055])
    fig.set_size_inches(4, 4)
    plt.tight_layout()
    
    return fig, ax


def build_arrow_list(env, policy):
    """Builds a list of arrows: [x, y, dx, dy] over the whole state space
    pointing in the direction indicated by the policy"""

    arrow_list = []
    scaling = 0.4 # arrow length scaling
   
    for x in np.arange(start=0., stop=1.01, step=0.1):
        for y in np.arange(start=0., stop=1.01, step=0.1):
            state = torch.tensor([[x, y]]).float()
            if x >= 0.6 or y >= 0.3: # skip states under legend box
                env.reset_to(state)
                with torch.no_grad():
                   action = policy.select_action(state)
                   
                next_state, _, _, _ = env.step(action)
                dx = scaling*(next_state - state)[0,0].item()
                dy = scaling*(next_state - state)[0,1].item()
                arrow_list.append([x, y, dx, dy])
               
    states = [(0.5, 0.2), (0.4, 0.2)]
    for (x,y) in states:
        state = torch.tensor([[x, y]]).float()
        env.reset_to(state)
        with torch.no_grad():
            action = policy.select_action(state)
           
        next_state, _, _, _ = env.step(action)
        dx = scaling*(next_state - state)[0,0].item()
        dy = scaling*(next_state - state)[0,1].item()
        arrow_list.append([x, y, dx, dy])
    
    return arrow_list


def plot_arrow_list(arrow_list):
    """Plots the policy arrow map showing the directions chosen by the policy
    based on a list of lists [x, y, dx, dy] for each arrow"""
 
    arrow_width = 0.008
    arrow_color = 'tab:blue' #'grey' # #'tab:orange'#'black'#
    arrow_edge_color = 'black'
    
    for i in range(len(arrow_list)):
        x, y, dx, dy = arrow_list[i]        
        if abs(y-0.7) < 0.01 and x >= 0.4 and dy > 0: # arrow violating constraint in red
            plt.arrow(x, y, dx, dy, width = arrow_width, edgecolor=arrow_edge_color,
                      facecolor="red", zorder=2)
        else:
            plt.arrow(x, y, dx, dy, width = arrow_width, edgecolor=arrow_edge_color,
                      facecolor=arrow_color, zorder=2)
    

def policy_map(env, policy, POLICEd=True, title = ''):
    """Plots the policy arrow map showing the directions chosen by the policy.
    Return a list of arrays [x, y, dx, dy] to plot the arrows"""
    
    arrow_list = build_arrow_list(env, policy)
    fig, ax = background(env, POLICEd)
    plot_arrow_list(arrow_list)   
    plt.title(title + " policy map")
    plt.show()
    return arrow_list
    
  


#%% Training utils

class ReplayBuffer(object):
    """Buffer to save experiences as (state, action, next_state, reward, not_done)"""
    def __init__(self, state_size, action_size, max_size=int(1e6), priority_method="", alpha=1):
        self.max_size = max_size
        self.ptr = 0 # id where next experience can be stored
        self.size = 0 # size of the buffer

        self.state = torch.zeros((self.max_size, state_size))
        self.action = torch.zeros((self.max_size, action_size))
        self.next_state = torch.zeros((self.max_size, state_size))
        self.reward = torch.zeros((self.max_size, 1))
        self.not_done = torch.zeros((self.max_size, 1))
        self.priority_method = priority_method
        self.alpha = alpha

        if self.priority_method == "reward":
            # Store minimal reward to make them all positive
            self.min_reward = np.inf
        
       
    def add(self, state, action, next_state, reward, done):
        """Adds an experience to the buffer."""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        if self.priority_method == "reward":
            if reward < self.min_reward:
                self.min_reward = reward
            
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        if(batch_size > self.size):
            print("The replay buffer does not have enough elements!")
            return
        # Redirect based on priority method, or just returns a uniformly sampled 
        if(self.priority_method == "index"):
            return self.index_priority_sample(batch_size)
        elif(self.priority_method == "reward"):
            return self.reward_priority_sample(batch_size)
        else:
            ind = np.random.randint(0, self.size, size=batch_size)
            return (self.state[ind].to(device), self.action[ind].to(device), self.next_state[ind].to(device), self.reward[ind].to(device), self.not_done[ind].to(device))

    def index_priority_sample(self, batch_size):
        possible_indices = np.arange(1,self.size+1)
        priorities = np.power(possible_indices, self.alpha)
        ind = np.random.choice(possible_indices, p=np.divide(priorities, priorities.sum()), size=batch_size)
        return (self.state[ind].to(device), self.action[ind].to(device), self.next_state[ind].to(device), self.reward[ind].to(device), self.not_done[ind].to(device))
    
    def reward_priority_sample(self, batch_size):
        possible_indices = torch.arange(self.size)
        priorities = torch.pow(self.reward[:self.size]-self.min_reward+1e-3, self.alpha).double().flatten()
        ind = np.random.choice(possible_indices, p=torch.div(priorities, priorities.sum()), size=batch_size)
        return (self.state[ind].to(device), self.action[ind].to(device), self.next_state[ind].to(device), self.reward[ind].to(device), self.not_done[ind].to(device))




class episode_stats_queue(object):
    """Queue to store the rewards of the last 100 episodes."""
    def __init__(self, max_size = int(100)):
        self.reward_queue = deque()
        self.size = 0
        self.max_size = max_size

    def add(self, episode_reward):
        """Adds an element to the end of the queue."""
        self.reward_queue.append(episode_reward)
        if self.size == self.max_size: # max size, pop first one, size doesn't change
            self.reward_queue.popleft()
        else: # not max size, increase it
            self.size += 1
            
    def average(self):
        """Calculates the average episode reward."""
        return sum(self.reward_queue)/self.size
            

