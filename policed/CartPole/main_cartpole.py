# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:55:08 2023

@author: Jean-Baptiste Bouvier

Safe Cartpole main with DDQN.

Goal: learn to stabilize the pole and guarantee that action is right push 
everywhere in the buffer where pole angle is too large.
With POLICEd, we only need to check whether the vertices of the buffer
have a highest Q-value for the right push action.
If there is a right push all over the buffer, then the pole can never fall
past the right angle boundary located at the right of the angle buffer.
"""

import torch
import numpy as np

from DDQN import DDQN
from safe_cartpole import Safe_CartPoleEnv
from utils import ReplayBuffer, episode_stats_queue, vertices
from utils import NetworkCopy, same_activation, local_affine_map, is_action_right_push

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Hyperparameters

POLICEd = True
# POLICEd = False
N_step = 200          # Length of each episode
N_episodes = 1000     # Number of training episodes

prb_alpha = 1         # Prioritized Experience Replay alpha value
prb_method = "reward" # Using prioritization by reward

lr = 1e-4          # learning rate
width = 128        # Number of units per hidden layer
batch_size = 128   # Batch size

freq_reset_in_buffer = 10 # reset 1 in x initial states into the buffer
reward_threshold = 0.8*N_step   # minimum average reward to consider the controller well trained,

### Environment
max_angle = 0.18 # 'None' or maximum pole angle

action_size = 1 # size of the action vector
N_actions = 2   # number of possible actions
state_size = 4  # size of the state vector

buffer = torch.tensor([[-1., 1.], # cart position
                        [-1., 1.], # cart velocity
                        [max_angle-0.08, max_angle],  # pole angle
                        [0., 1.]]) # pole velocity
buffer_vertices = vertices(buffer)

### Penalizing left push in buffer
env = Safe_CartPoleEnv(max_angle)

### Initialize policy
controller = DDQN(state_size, N_actions, width, POLICEd, batch_size, lr, buffer)
replay_buffer = ReplayBuffer(state_size, action_size, max_size=N_episodes*N_step, priority_method=prb_method, alpha=prb_alpha)


  
  
#%% Training

stats_queue = episode_stats_queue(max_size = 100)
trained = False

for episode in range(1, N_episodes):
        
    if episode % freq_reset_in_buffer == 0:
        ### Random sampling in buffer
        coefs = torch.rand((1, buffer_vertices.shape[0]))
        coefs /= coefs.sum(dim=1)
        state = coefs @ buffer_vertices
        env.reset_to_state(state)
    else:  
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_reward = 0.

    for t in range(N_step):
        
        ### Select action with epsilon greedy policy
        action = controller.select_action(env, state)
    
        ### Perform action
        observation, reward, done, _, _ = env.step(action.item())#.to(device)
        episode_reward += reward
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        ### Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, float(done))
        state = next_state
    
        ### Train agent after collecting sufficient data
        if replay_buffer.size > batch_size:
            controller.train(replay_buffer)
        if done: break
    
    stats_queue.add(episode_reward)
    average_reward = stats_queue.average()
    
    if POLICEd and average_reward > reward_threshold:
        controller.policy.enlarge_buffer = True
    
    ### Evaluate episodes
    if episode % 100 == 0:
        print(f"Episode: {episode} \t Average reward: {average_reward:.2f}"),
        is_right = is_action_right_push(env, controller, display=True)
        trained = average_reward > reward_threshold and is_right
        if(trained):
            break
        
    elif episode % 10 == 0:
        if POLICEd and controller.policy.iter < controller.policy.max_iter:
            continue
        is_right = is_action_right_push(env, controller, display=False)
        trained = average_reward > reward_threshold and is_right
        if(trained):
            break
        
print(f"Episode: {episode} \t Average reward: {average_reward:.2f} \n")




#%% Visualisation

import matplotlib.pyplot as plt

env = Safe_CartPoleEnv(max_angle, render_mode="human")  
state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
episode_reward = 0.
cart_position = np.zeros(N_step)
pole_angle = np.zeros(N_step)
   
for t in range(N_step):
    with torch.no_grad():
        action = controller.max_likelihood_action(state)
    observation, reward, done, _, _ = env.step(action.item())
    episode_reward += reward
    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
    cart_position[t] = state[0,0]
    pole_angle[t] = state[0,2] 
    state = next_state
    
    env.render()
    if done: break

print(f"Average reward: {episode_reward/N_step:.3f}")
plt.title("Cartpole position")
plt.scatter(np.arange(t+1), cart_position[:t+1], s=10)
plt.plot(np.array([0, N_step]),  np.ones(2)*env.x_threshold, c='red')
plt.plot(np.array([0, N_step]), -np.ones(2)*env.x_threshold, c='red')
plt.show()

plt.title("Pole angle (rad)")
plt.scatter(np.arange(t+1), pole_angle[:t+1], s=10, label="angle")
plt.plot(np.array([0, N_step]),  np.ones(2)*env.theta_threshold_radians, c='red')
plt.plot(np.array([0, N_step]), -np.ones(2)*env.theta_threshold_radians, c='red', label="limit")
if max_angle is not None:
    plt.plot(np.array([0, N_step]), np.ones(2)*max_angle, c='purple', label="constraint", linewidth=4)
    plt.fill(np.array([0, N_step, N_step, 0]), np.array([buffer[2,0], buffer[2,0], buffer[2,1], buffer[2,1]]), c="cyan", alpha=0.2, label="buffer")
plt.legend()
plt.show()


#%% End visualisation

env.close()

#%% POLICEd verifications

env.render_mode = None



def norm(s):
    return torch.linalg.vector_norm(s)

if POLICEd:
    copied = NetworkCopy(controller.policy)
    same_activation(copied)
    C, d = local_affine_map(copied, buffer_vertices[0])
    eps = 1e-3
    
    ### Verification that the affine map is the same everywhere
    for vertex_id in range(buffer_vertices.shape[0]):
        s = buffer_vertices[vertex_id].unsqueeze(dim=0)
        with torch.no_grad():
            copy = copied(s)
            tru = controller.policy(s)
        assert norm(copy - tru)/norm(tru) < eps, "Network copy does not match policy" 
        ### Normalization is messing up the POLICEd verification
        assert norm(s @ C.T + d - tru)/norm(tru) < eps, "Affine map does not match policy"
    
### Instead, we check that the action is a right push all over the buffer,
### so that the cart should not reach the left wall
is_right = is_action_right_push(env, controller)


   
        






