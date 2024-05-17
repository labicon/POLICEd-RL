"""
Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
https://github.com/sfujim/TD3
Paper: https://arxiv.org/abs/1802.09477

TD3 is augmented with the ConstrainedLayers from POLICE guaranteeing an affine
output in a given input region
POLICE code from  https://github.com/RandallBalestriero/POLICE
Paper: https://arxiv.org/abs/2211.01340

TD3 is adapted to the linear environment with pytorch tensors
Choose in between regular policy 'Actor' and 
a constrained one with POLICE algorithm 'PolicedActor'.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def enforce_constraint_forward(x, W, b, C):
    """Perform a forward pass on the given `x` argument which contains both the `C` vertices
    describing the region `R` onto which the DNN is constrained to stay affine, and the mini-batch.
    Modified from POLICE"""
    h = x @ W.T + b
    V = h[-C:]
    with torch.no_grad():
        agreement = V > 0
        invalid_ones = agreement.all(0).logical_not_().logical_and_(agreement.any(0))
        sign = agreement[:, invalid_ones].half().sum(0).sub_(C / 2 + 1e-3).sign_()

    extra_bias = (V[:, invalid_ones] * sign - 1e-4).amin(0).clamp(max=0) * sign
    h[:, invalid_ones] -= extra_bias
    return h


class ConstrainedLayer(nn.Linear):
    """Modified from POLICE"""
    def forward(self, x, C):
        return enforce_constraint_forward(x, self.weight, self.bias, C)


class PolicedActor(nn.Module):
    def __init__(self, env, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.nb_layers = 3
        self.constraint_area = env.allowed_constraint_area
        
        # First layer
        self.layer0 = ConstrainedLayer(env.state_size, hidden_size)
        # Hidden layers
        for i in range(1, self.nb_layers-1):
            setattr(self, f"layer{i}", ConstrainedLayer(hidden_size, hidden_size) )
        # Last layer
        setattr(self, f"layer{self.nb_layers-1}", ConstrainedLayer(hidden_size, env.action_size) )
    
    def forward(self, x):
        with torch.no_grad():
            x = torch.cat( (x, self.constraint_area), dim=0)
        C = self.constraint_area.shape[0]
        for i in range(self.nb_layers-1):
            x = getattr(self, f"layer{i}")(x, C)
            x = torch.relu(x)
        x = getattr(self, f"layer{self.nb_layers-1}")(x, C)
        return x[:-C] # unbounded output
        
        


class Actor(nn.Module):
    def __init__(self, env, hidden_size):
        super(Actor, self).__init__()

        self.net = nn.Sequential(nn.Linear(env.state_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, env.action_size))
        
    def forward(self, state):
        return self.net(state) # unbounded controller


class Critic(nn.Module):
    def __init__(self, env, hidden_size):
        super(Critic, self).__init__()
        self.Q1_net = nn.Sequential(nn.Linear(env.state_size + env.action_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, 1))
        self.Q2_net = nn.Sequential(nn.Linear(env.state_size + env.action_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.Q1_net(sa), self.Q2_net(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.Q1_net(sa)


class TD3(object):
    def __init__(self, env, POLICEd, hidden_size, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        if POLICEd:
            self.actor = PolicedActor(env, hidden_size).to(device)
        else:
            self.actor = Actor(env, hidden_size).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=1e-5)

        self.critic = Critic(env, hidden_size).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-5)

        self.action_min = env.action_min
        self.action_max = env.action_min
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        return (self.actor(state.to(device))).cpu()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
			# Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)#.clamp(self.action_min, self.action_max)

			# Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

		# Delayed policy updates
        if self.total_it % self.policy_freq == 0:

			# Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

			# Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
		