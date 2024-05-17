# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:09:16 2023

@author: Jean-Baptiste Bouvier

Simple PyTorch environment for the 2D linear system.
"""

import copy
import torch
import numpy as np


class LinearEnv():
    """2D Linear system environment"""
    
    def __init__(self, target = torch.tensor([[0.9, 0.9]]),
                       constraint_min = torch.tensor([[0.4, 0.7]]),
                       constraint_max = torch.tensor([[1., 0.7]])):
        ### Hyperparameters
        self.dt = 0.1 # time step
        self.state_size = 2 # state dimension
        self.action_size = 2 # input dimension
        self.action_min = -torch.ones((1, self.action_size))
        self.action_max = torch.ones((1, self.action_size))
        
        ### Linear dynamics: dx/dt = Ax + Bu
        self.A = -0.1*torch.eye((self.state_size))
        self.B = torch.eye((self.state_size))
        
        ### State space limits (square [0,1] x [0,1])
        self.x_min = torch.tensor([[0., 0.]])
        self.x_max = torch.tensor([[1., 1.]])
        
        ### Forbidden to cross the line between constraint_min and constraint_max
        self.constraint_min = constraint_min
        self.constraint_max = constraint_max
        self.width_affine_area = torch.tensor([[0., -0.1]])

        self._allowed_constraint_area()
        self.constraint_line_x1 = np.array([constraint_min[0,0], constraint_max[0,0]])
        self.constraint_line_x2 = np.array([constraint_min[0,1], constraint_max[0,1]])
        
        ### Target
        self.target = target
        self.precision = 0.05
        
        
          
    def reward(self, state):
        """Reward is the distance to the target"""
        return -torch.linalg.vector_norm(state - self.target, dim=1)
        
    def respect_constraint(self, new_state):
        """Checks whether new_state respects the constraint."""
        return not crossing(self.constraint_min, self.constraint_max, self.state, new_state)
    
    
    def arrived(self):
        """Episode is over when target is reached within given precision."""
        return torch.linalg.vector_norm(self.target - self.state).item() < self.precision
    
    def _propagation(self, u):
        """Propagates the dynamics one step forward for control input u."""
        new_state = self.state + self.dt * (self.state @ self.A + u @ self.B)
        respect = self.respect_constraint(new_state)        
        too_far = (new_state < self.x_min).any().item() or (new_state > self.x_max).any().item()
        return new_state, respect, too_far
        
            
    def step(self, u):
        """Returns: new_state, reward, done, respect"""
        self.state, respect, too_far = self._propagation(u)
        done = self.arrived() or too_far or not respect
        reward = self.reward(self.state).item() + done -3*(not respect) -3*too_far
        reward -= 20*torch.relu(u.abs().max(dim=1).values -1).item() # penalize |u| > 1
        return copy.deepcopy(self.state), reward, done, respect


    def training_step(self, u):
        """Takes a step in the environment. If the constraint is violated or if
        the states exits the admissible area, the state is maintained in place
        and a penalty is assigned. This prevents too short trajectories.
        'done' is only assigned if the target is reached."""
        
        new_state, respect, too_far = self._propagation(u)
        ### Update to new state only if constraint is not violated and hasn't exited allowed area
        if respect and not too_far:
            self.state = new_state
        done = self.arrived()
        reward = self.reward(self.state).item() + done -3*(not respect) -3*too_far
        reward -= 20*torch.relu(u.abs().max(dim=1).values -1).item() # penalize |u| > 1
        return copy.deepcopy(self.state), reward, done, respect
        

    def reset(self, max_dist_target = 2):
        """Resets the state of the linear system to a random state in [x_min, x_max]."""
        self.state = (self.x_max - self.x_min) * torch.rand(1, self.state_size) + self.x_min
        while torch.norm(self.state - self.target) > max_dist_target:
            self.state = (self.x_max - self.x_min) * torch.rand(1, self.state_size) + self.x_min
        return copy.deepcopy(self.state)
    
    def reset_in_constraint_area(self):
        """Resets the state of the linear system to a random state in the allowed constraint area."""
        N_vertices = self.allowed_constraint_area.shape[0]
        coefs = torch.rand(1, N_vertices)
        coefs/= coefs.sum()
        self.state = coefs @ self.allowed_constraint_area
        return copy.deepcopy(self.state)
    
    
    def reset_to(self, state):
        """Resets the state of the linear system to a given state."""
        self.state = copy.deepcopy(state)
        return self.state
    
    def random_action(self):
        """Returns a uniformly random action within action space."""
        return (self.action_max - self.action_min) * torch.rand(1, self.action_size) + self.action_min

             
    def _allowed_constraint_area(self):
        """Calculates where it is allowed."""
        self.allowed_constraint_area = torch.cat((self.constraint_min,
                                                  self.constraint_max, 
                                                  self.constraint_max + self.width_affine_area,
                                                  self.constraint_min + self.width_affine_area))
        
   
                

##################################################################
################## Crossing of segments ##########################
##################################################################
# Based on https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect


def cross_product(v, w):
    """Returns the 2D cross-product of two 2D vectors."""
    return v[0]*w[1] - v[1]*w[0]

def crossing(a0, a1, b0, b1):
    """Verifies whether 2D segment [a0, a1] crosses with 2D segment [b0, b1]."""
    p = a0.squeeze()
    q = b0.squeeze()
    r = (a1 - a0).squeeze()
    s = (b1 - b0).squeeze()
    rs = cross_product(r, s)
    
    if rs == 0: ### parallel
        cross = torch.tensor(False)
    else: ### not parallel
        t = cross_product(q-p, s)/rs
        u = -cross_product(p-q, r)/rs
        cross = (0 <= t) and (t <= 1) and (0 <= u) and (u <= 1)
    
    return cross.item()               
                
                
                
            