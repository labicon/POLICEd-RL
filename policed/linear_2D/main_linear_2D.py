# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:09:16 2023

@author: Jean-Baptiste Bouvier

Main for POLICEd RL applied on the 2D linear system.
"""


import time, math
import torch
import numpy as np

from TD3 import TD3
from linear_env import LinearEnv
from utils import ReplayBuffer, episode_stats_queue
from utils import policy_map, empirical_constraint_violation
from utils import NetworkCopy, local_affine_map, same_activation, activation_map
 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Chooses whether the policy is POLICEd or not, i.e., whether it is affine in front of constraint
POLICEd = True
### Chooses whether to load a trained POLICEd model
load = False # True 
filename = "trained_model/POLICEd"



#%% Hyperparameters

N_step = 128                    # Length of each episode
start_episode = 100             # Number of episodes where random policy is used

prb_alpha = 1                   # Prioritized Experience Replay alpha value

policy_freq = 2                 # Frequency in episodes of delayed policy updates
in_constraint_freq = 20         # Frequency in episodes at which the initial state is chosen in the allowed constraint area

hidden_size = 64                # Number of units per hidden layer
batch_size = 256                # Batch size for both actor and critic
discount = 0.99                 # Discount factor
tau = 0.004                     # Target network update rate

reward_threshold = -10          # minimum average reward to consider the controller well trained
buffer_priority_method = "reward"

### Cornered target
constraint_min = torch.tensor([[0.4, 0.7]])
constraint_max = torch.tensor([[1.0, 0.7]])
target = torch.tensor([[0.9, 0.9]])


env = LinearEnv(target, constraint_min, constraint_max)     # Linear Environment that you defined

expl_noise = 0.1*(env.action_max - env.action_min)/2        # Std of Gaussian exploration noise
policy_noise = 0.1*(env.action_max - env.action_min)/2      # Noise added to target policy during critic update
noise_clip = 0.2*(env.action_max - env.action_min)/2        # Range to clip target policy noise



#%% Initialize policy

controller = TD3(env, POLICEd, hidden_size, discount, tau, policy_noise, noise_clip, policy_freq)
if load:
    controller.load(filename)
else:
    replay_buffer = ReplayBuffer(env.state_size, env.action_size, max_size=1000*N_step, priority_method=buffer_priority_method, alpha=prb_alpha)
    replay_buffer.sample(batch_size)
     
    
    #%% Training
    t0 = time.time()
    
    stats_queue = episode_stats_queue(max_size = 100)
    trained = False
    episode = 100
    for episode in range(10000):
        if episode % in_constraint_freq == 0: # regularly start state in allowed constraint area
            state = env.reset_in_constraint_area()
        else: # otherwise initial state is in a growing area centered on the target
            state = env.reset(max_dist_target = 0.2 + episode/800)
        episode_reward = 0
    
        for t in range(N_step):
            ### Select action randomly or according to policy
            if episode < start_episode:
                action = env.random_action()
            else:
                with torch.no_grad():
                    action = controller.select_action(state)
                action += torch.randn_like(action) * expl_noise * math.exp(-1. * episode/400)
                
            ### Perform action
            next_state, reward, done, respect = env.training_step(action)
            ### Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, float(done))
            state = next_state
            episode_reward += reward
           
            ### Train agent after collecting sufficient data
            if episode >= start_episode:
                controller.train(replay_buffer, batch_size)
            if done: break
    
        stats_queue.add(episode_reward)
        average_reward = stats_queue.average()
        ### Evaluate episodes
        if episode % 100 == 0:
            print(f"Episode: {episode} \t Average reward: {average_reward:.2f}")
            violation = empirical_constraint_violation(env, controller, 1000, True)
            trained = average_reward > reward_threshold and violation == 0.
            policy_map(env, controller, POLICEd, title=f"Episode {episode}:")
            if(trained):
                break
        elif episode % 10 == 0:
            violation = empirical_constraint_violation(env, controller, 100, False)
            trained = average_reward > reward_threshold and violation == 0.
            if(trained):
                break

    
    print(f"Episode: {episode} \t Average reward: {average_reward:.2f}\n")
    policy_map(env, controller, POLICEd, title=f"Episode {episode}:")
    empirical_constraint_violation(env, controller, 1000, True)
    
    t1 = time.time()
    print(f"elapsed time = {(t1-t0)/60:.0f} min or {(t1-t0):.5f} s")
   

#%% Verifying

if POLICEd:
    ### Copying policy network with POLICE extra-bias incorporated
    policy = NetworkCopy(controller.actor)
    
    ### Points in constraint area
    X = controller.actor.constraint_area
    
    ### Verifying success of network copy
    with torch.no_grad():
        U = controller.actor(X)
        U_copy = policy(X)
        
    print(f"Difference between network and its copy on vertices: ||U - U_copy|| = {torch.norm(U - U_copy).item():.3f}")
    
    nb_vertices = X.shape[0]
    C_matrices = torch.zeros((nb_vertices, env.action_size, env.state_size)).to(device)
    d_matrices = torch.zeros((nb_vertices, env.action_size)).to(device)
    
    ### Verifying accuracy of affine maps
    for i in range(nb_vertices):
        vertex = X[i]
        C_matrices[i], d_matrices[i] = local_affine_map(policy, vertex)
        print(f"Difference between NN and affine map at [{vertex[0].item():.2f}, {vertex[1].item():.2f}]: ||C @ x + d - u|| = {torch.norm(vertex @ C_matrices[i].T + d_matrices[i] - U[i]).item():.3f}")
    
    ### Verifying similarity of affine map
    same_map = True
    for i in range(nb_vertices-1):
        Ci, di = C_matrices[i], d_matrices[i]
        for j in range(i+1, nb_vertices):
            Cj, dj = C_matrices[j], d_matrices[j]
            if torch.norm(Ci - Cj) + torch.norm(di - dj) > 1e-6:
                same_map = False
                print(f"Affine maps are different at vertices {i} and {j} ||Ci - Cj|| = {torch.norm(Ci - Cj).item():.6f}  ||di - dj|| = {torch.norm(di - dj).item():.6f}")
    if same_map:
        print("The affine map is constant on all vertices.")

    same_activation(policy)
    activation_map(policy, env)




#%% Saving the controller

# controller.save(filename)







#%% Final plot

import matplotlib.lines as mlines
import matplotlib.pyplot as plt


target_color = 'cyan'
constraint_color = 'red'
buffer_color = (144./255, 238./255, 144./255)

fig = plt.gcf()
ax = fig.gca()

plt.rcParams.update({'font.size': 14})
plt.rcParams['font.sans-serif'] = ['Palatino Linotype']

### Buffer
buffer = plt.Rectangle(xy=[0.36, 0.6], width=0.66, height=0.11, color=buffer_color)
ax.add_patch(buffer)


scaling = 0.4 # arrow length scaling
arrow_width = 0.008
arrow_color = 'tab:blue' 
arrow_edge_color = 'black'
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
            plt.arrow(x, y, dx, dy, width = arrow_width,
                      edgecolor=arrow_edge_color, facecolor=arrow_color)


states = [(0.5, 0.2), (0.4, 0.2)]
for (x,y) in states:
    state = torch.tensor([[x, y]]).float()
    env.reset_to(state)
    
    with torch.no_grad():
        action = policy.select_action(state)
       
    next_state, _, _, _ = env.step(action)
    dx = scaling*(next_state - state)[0,0].item()
    dy = scaling*(next_state - state)[0,1].item()
    plt.arrow(x, y, dx, dy, width = arrow_width, edgecolor=arrow_edge_color,
              facecolor=arrow_color)

### Target
target_circle = plt.Circle((env.target[0,0].item(), env.target[0,1].item()), env.precision, color=target_color, zorder=2)
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
buffer_marker = mlines.Line2D([], [], color=buffer_color, marker='s', linestyle='None', markersize=10, label='affine buffer $\mathcal{B}$')
constraint_marker = mlines.Line2D([], [], color=constraint_color, linewidth='4', markersize=10, label='constraint line')

plt.legend(handles=[target_marker, constraint_marker, buffer_marker],
           loc='lower left', frameon=False, borderpad=.0,
           labelspacing=0.3, handletextpad=0.5, handlelength=1.4)


ax.set_yticks([0.3, 0.5, 0.7, 0.9])
ax.set_xticks([0., 0.2, 0.4, 0.6, 0.8, 1.0])
plt.xlim([-0.02, 1.055])
plt.ylim([-0.02, 1.055])

fig.set_size_inches(4, 4)
plt.show()
