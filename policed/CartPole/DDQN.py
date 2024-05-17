# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:58:42 2023

@author: Jean-Baptiste BOUVIER

POLICEd DDQN for environments with discrete actions

Implementation of Double Deep Q-network (DDQN)
Paper: https://arxiv.org/abs/1509.06461

DDQN is augmented with the ConstrainedLayers from POLICE guaranteeing an affine
output in a given input region
POLICE code from  https://github.com/RandallBalestriero/POLICE
Paper: https://arxiv.org/abs/2211.01340
"""

import math
import copy
import torch
import random
import torch.nn as nn
from utils import vertices


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enforce_constraint_forward(x, W, b, num_vertices):
    """Perform a forward pass on the given `x` argument which contains both the `num_vertices` vertices
    describing the region `R` onto which the DNN is constrained to stay affine, and the mini-batch"""
    h = x @ W.T + b
    V = h[-num_vertices:]
    with torch.no_grad():
        agreement = V > 0
        invalid_ones = agreement.all(0).logical_not_().logical_and_(agreement.any(0))
        sign = agreement[:, invalid_ones].half().sum(0).sub_(num_vertices / 2 + 0.01).sign_()

    extra_bias = (V[:, invalid_ones] * sign - 1e-4).amin(0).clamp(max=0) * sign
    h[:, invalid_ones] -= extra_bias
    return h


class ConstrainedLayer(nn.Linear):
    def forward(self, x, num_vertices):
        return enforce_constraint_forward(x, self.weight, self.bias, num_vertices)


class POLICEdPolicy(nn.Module):
    def __init__(self, state_size, action_size, width, buffer):
        super().__init__()
        self.width = width
        self.nb_layers = 3
        
        ### First layer
        self.layer0 = ConstrainedLayer(state_size, width)
        ### Hidden layers
        for i in range(1, self.nb_layers-1):
            setattr(self, f"layer{i}", ConstrainedLayer(width, width) )
        ### Last layer
        setattr(self, f"layer{self.nb_layers-1}", ConstrainedLayer(width, action_size) )
    
        ### Affine buffer
        self.buffer_lb = buffer[:,[0]] # lower bound of the buffer
        self.buffer_ub = buffer[:,[1]] # upper bound of the buffer
        ### Initial buffer of volume 0
        self.buffer_vertices = vertices(torch.cat((self.buffer_lb, self.buffer_lb), dim=1))
        self.num_vertices = self.buffer_vertices.shape[0]
        ### Counting iterations to slowly enlarge the buffer once allowed
        self.iter = 0
        self.max_iter = 10000
        ### Initially not allowed to enlarge the buffer
        self.enlarge_buffer = False
    
    def enlarging_buffer(self):
        """Iteratively enlarges the affine buffer until reaching the desired volume."""
        self.iter += 1
        if self.iter > self.max_iter:
            self.enlarge_buffer = False
        
        if self.iter % 10 == 0:
            ub = self.buffer_lb + (self.buffer_ub - self.buffer_lb) * self.iter/self.max_iter
            buffer = torch.cat((self.buffer_lb, ub),dim=1)
            self.buffer_vertices = vertices(buffer)
    
    def forward(self, x):
        if self.enlarge_buffer:
            self.enlarging_buffer()
            
        with torch.no_grad():
            x = torch.cat( (x, self.buffer_vertices), dim=0)
        for i in range(self.nb_layers-1):
            x = getattr(self, f"layer{i}")(x, self.num_vertices)
            x = torch.relu(x)
        x = getattr(self, f"layer{self.nb_layers-1}")(x, self.num_vertices)
        return x[:-self.num_vertices] # unconstrained controller





class Policy(nn.Module):
    def __init__(self, state_size, action_size, width):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(state_size, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, action_size))
       
    def forward(self, state):
        return self.net(state) # unconstrained controller
          
   
    
class DDQN(nn.Module):
    """Double DQN (has a target Q-network)"""
    def __init__(self, state_size, N_actions, width=64, POLICEd=False, batch_size=128, lr=1e-4,
                 buffer = torch.tensor([0., 0.])):
        super(DDQN, self).__init__()
       
        self.tau = 0.005 # update rate of the target network
        self.lr = lr  # learning rate of the ``AdamW`` optimizer 
        self.batch_size = batch_size
        self.gamma = 0.99 # discount factor
        self.target_update_frequency = 2
        
        ### Epsilon greedy policy
        self.steps_done = 0
        self.eps_start = 0.9 # EPS_START is the starting value of epsilon
        self.eps_end = 0.05 # EPS_END is the final value of epsilon
        self.eps_decay = 5000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        
        self.N_actions = N_actions # number of possible actions
        self.state_size = state_size
        
        self.buffer = buffer
        self.buffer_vertices = vertices(buffer)
        self.POLICEd = POLICEd
        
        if POLICEd:
            self.policy = POLICEdPolicy(state_size, N_actions, width, self.buffer).to(device)
        else:
            self.policy = Policy(state_size, N_actions, width).to(device)
            
        self.policy_target = copy.deepcopy(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4, weight_decay=1e-3)
    
    def select_action(self, env, state):
        """Epsilon greedy action selection."""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end)*math.exp(-self.steps_done/self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    def max_likelihood_action(self, state):
        """Return action with highest likelihood."""
        return self.policy(state).max(1)[1].view(1, 1)

    def train(self, replay_buffer):
       
        ### Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
        
        ### Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy
        action = action.type(torch.int64) 
        state_action_values = self.policy(state).gather(1, action)

        ### Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        not_done = not_done.squeeze().bool()
        non_final_next_states = next_state[not_done]
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[not_done] = self.policy_target(non_final_next_states).max(1)[0]
        ### Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward

        ### Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        ### Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        ### In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()
        
        ### Delayed target updates
        if self.steps_done % self.target_update_frequency == 0:
            ### Soft update of the target network's weights
            ### θ′ ← τ θ + (1 −τ )θ′
            for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


            
            
            
