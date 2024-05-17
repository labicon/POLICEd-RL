# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:55:08 2023

@author: Jean-Baptiste Bouvier

Inverted Pendulum main with PPO
Goal: learn to stabilize the pole and guarantee that

PPO implementation from
https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous

"""

import torch
import argparse
import numpy as np

from PPO import PPO
from safe_invPendulum import Safe_InvertedPendulumEnv
from utils import ReplayBuffer, buffer_vertices, epsilon, plot_traj, data
from utils import constraint_verification, training, constraint_training
from normalization import Normalization, RewardScaling


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%% Hyperparameters


parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
parser.add_argument("--max_train_steps", type=int, default=int(1e6), help="Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Gaussian") # or Beta")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=32, help="Minibatch size")
parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2: state normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4: reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6: learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--seed", type=int, default=10, help="Common seed for all environments")

### POLICEd additions
parser.add_argument("--POLICEd", type=bool, default=True, help= "Whether the layers are POLICEd or linear")
parser.add_argument("--enlarging_buffer", type=bool, default=True, help= "Whether the buffer starts from a volume 0 and increases once policy has converged")
parser.add_argument("--max_iter_enlargment", type=int, default=int(1e4), help="Number of enlarging iterations for the buffer")
b = 0.1
parser.add_argument("--constraint_C", type=np.array, default=np.array([0., b, 0., 1.]), help="C matrix for the constraint  C @ state < d")
parser.add_argument("--constraint_d", type=float, default=0., help="d matrix for the constraint  C @ state < d")
parser.add_argument("--min_state", type=np.array, default=np.array([-0.9, 0.1, -1., 0.]), help="min value for the states in the buffer")
parser.add_argument("--max_state", type=np.array, default=np.array([0.9, 0.2, 1., 0.]), help="max value for the states in the buffer")
args = parser.parse_args()

args.enlarging_buffer = False
# args.POLICEd = False

### Set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

### Environment
env = Safe_InvertedPendulumEnv(args, dense_reward=True)
# When evaluating the policy, we need to rebuild an environment
env_eval = Safe_InvertedPendulumEnv(args, dense_reward=False)  

### Constraint and buffer
buffer_vertices, args.r = buffer_vertices(args, env_eval, display=True)
args.buffer_vertices = buffer_vertices
args.min_state[-1] = -args.r
args.eps = epsilon(args, env_eval)
args.tol = -2*args.eps*env.dt

### Add new constants to the environments
env.tol = args.tol
env_eval.tol = args.tol
env.r = args.r
env_eval.r = args.r

### Add new constants to the arguments
args.state_dim = env.state_size
args.action_dim = env.action_size
args.max_action = env.action_max

agent = PPO(args)
replay_buffer = ReplayBuffer(args)

### state normalization
state_norm = Normalization(shape=args.state_dim)  
training_data = data()

if args.use_reward_scaling:  # reward scaling
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

args.max_episode = 1500


#%% Training
trained = False
stable = False

# while not trained:
while training_data.len < args.max_episode:
    if not stable:
        stable = training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling, training_data)

    stable, respect = constraint_training(args, env, agent, replay_buffer, env_eval, state_norm, reward_scaling, training_data)
    trained = stable and respect
            

#%% Visualisation 

env = Safe_InvertedPendulumEnv(args, dense_reward=False, render_mode="human")  
state = env.reset()

episode_reward = 0.
Trajectory = np.zeros((env.max_episode_steps, env.state_size))
Trajectory[0] = state
Actions = np.zeros((env.max_episode_steps, env.action_size))

for t in range(env.max_episode_steps):
    with torch.no_grad():
        action = agent.evaluate(state, state_norm)
    state, reward, done, _ = env.step(action)
    episode_reward += reward
    
    Trajectory[t] = state
    Actions[t-1] = action
    
    env.render()
    if done: break

print(f"Average reward: {episode_reward/env.max_episode_steps:.3f}")
plot_traj(env, Trajectory[:t+1], env.max_episode_steps)

print(f"Min max actions on visual traj:  {Actions.min().item():.2f},  {Actions.max().item():.2f}")


#%% End visualisation
env.close()

  
#%% POLICEd verifications

env.render_mode = None

from utils import NetworkCopy, same_activation, local_affine_map

def norm(s):
    return torch.linalg.vector_norm(s).item()


if args.POLICEd:
    copied = NetworkCopy(agent.actor, state_norm)    
    same_activation(copied)
    C, d = local_affine_map(copied, copied.buffer_vertices[0]) # already normalized vertices
    eps = 1e-3
    
    ### Verification that the affine map is the same everywhere
    for vertex_id in range(buffer_vertices.shape[0]):
        s = torch.tensor(buffer_vertices[vertex_id]).float().unsqueeze(dim=0)
        normalized_s = state_norm(s, update=False) # normalized state
        with torch.no_grad():
            copy = copied(s)
            tru = agent.actor(s, state_norm, update=False)
        assert norm(copy - tru)/norm(tru) < eps, "Network copy does not match policy" 
        ### Normalization is messing up the POLICEd verification
        assert norm(normalized_s @ C.T + d - tru)/norm(tru) < eps, "Affine map does not match policy"
    print("The original policy, the copy and the affine map all match on the buffer")
    

#%% Testing constraint_verification
 
constraint_verification(args, env_eval, agent, state_norm, display=True)

