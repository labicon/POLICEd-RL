# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:52:25 2023

@author: Jean-Baptiste Bouvier

Function utils for DDQN POLICEd applied on the CartPole environmnent.
"""

import copy
import torch
import itertools
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#%% Utils to verify that the POLICEd policy is affine on the given input region
    
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
        self.width = policy.width
        self.buffer_vertices = policy.buffer_vertices.clone()
        self.C = self.buffer_vertices.shape[0]
        self.setup(policy)
        
    def setup(self, policy):
        # Copy all layers from policy
        x = self.buffer_vertices
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
        """Returns action with the highest Q-value at given state."""
        Q = (self.forward(state.to(device))).cpu()
        return Q.max(1)[1].view(1,1)
    
    


def same_activation(policy):
    """Verifies whether the whole CONVEX constraint area has the same activation pattern.
    Returns a boolean and a vertex where activation differs."""
    
    nb_vertices = policy.buffer_vertices.shape[0] # number of vertices of the constraint area
    ### Verify that all vertices of the constraint area have the same activation pattern
    prev_AP = activation_pattern(policy, policy.buffer_vertices[0])
    for vertex_id in range(1,nb_vertices):
        vertex = policy.buffer_vertices[vertex_id].unsqueeze(dim=0)
        AP = activation_pattern(policy, vertex)
        if not (AP == prev_AP).all(): 
            print("The constraint region does not share the same activation pattern.")
            return False, vertex
        
    print("The constraint region shares the same activation pattern.")
    return True, vertex
     

def is_action_right_push(env, controller, display=True):
    """Verifies whether the action is push to the right on the whole buffer area."""
    
    nb_right_push = 0
    test_points = controller.buffer_vertices
    nb_test_points = test_points.shape[0]
    
    for vertex_id in range(nb_test_points):
        vertex = test_points[vertex_id].unsqueeze(dim=0)
        
        with torch.no_grad():
            action = controller.max_likelihood_action(vertex)
        if action.item() == 1:
            nb_right_push += 1
            
    if display:
        print(f"Action is right push at {100*nb_right_push/nb_test_points:.0f}% of the buffer vertices.")
    return nb_right_push == nb_test_points


#%% Other utils



def vertices(C):
    """Calculates all the vertices corresponding to the hyperrectangle defined by C."""
    dim = C.shape[0]
    assert C.shape[1] == 2, "C must be a tensor of ranges"
    
    L = list(itertools.product(range(2), repeat=dim))
    v = torch.zeros(2**dim, dim)
    rows = torch.arange(dim)
    
    for i in range(2**dim):
        cols = torch.tensor(L[i])
        v[i] = C[rows, cols]
        
    return v










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
        if(batch_size > self.size): # Quick Sanity Check!
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
        priorities = torch.pow(self.reward[:self.size], self.alpha).double().flatten()+0.1
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
            






  