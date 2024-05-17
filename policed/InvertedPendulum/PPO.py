"""
PPO implementation from
https://github.com/Lizhi-sjtu/DRL-code-pytorch/tree/main/5.PPO-continuous

Added the POLICEd layers, with a change:
    POLICE code from  https://github.com/RandallBalestriero/POLICE
    Paper: https://arxiv.org/abs/2211.01340
   
in function 'enforce_constraint_forward'
sign = agreement[:, invalid_ones].float().sum(0).sub_(C/2 + 1e-3).sign_()
replaces
sign = agreement[:, invalid_ones].half().sum(0).sub_(C/2 + 1e-3).sign_() 
as the half() was reducing the precision and the sign could end up being 0,
which renders this function ineffective.

"""


import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Normal



def enforce_constraint_forward(x, W, b, C):
    """Perform a forward pass on the given `x` argument which contains both the `C` vertices
    describing the region `R` onto which the DNN is constrained to stay affine, and the mini-batch"""
    h = x @ W.T + b
    V = h[-C:]
    with torch.no_grad():
        agreement = V > 0
        invalid_ones = agreement.all(0).logical_not_().logical_and_(agreement.any(0))
        sign = agreement[:, invalid_ones].float().sum(0).sub_(C/2 + 1e-3).sign_()

    extra_bias = (V[:, invalid_ones] * sign - 1e-4).amin(0).clamp(max=0) * sign
    h[:, invalid_ones] -= extra_bias
    return h


class ConstrainedLayer(nn.Linear):
    def forward(self, x, C):
        return enforce_constraint_forward(x, self.weight, self.bias, C)



### With buffer starting from 0 and progressively enlarging
class Enlarging_POLICEd_Gaussian_Actor(nn.Module):
    def __init__(self, args):
        super(Enlarging_POLICEd_Gaussian_Actor, self).__init__()
        self.nb_layers = 3
        self.width = args.hidden_width
        self.layer_0 = ConstrainedLayer(args.state_dim, self.width)
        self.layer_1 = ConstrainedLayer(self.width, self.width)
        self.layer_2 = ConstrainedLayer(self.width, args.action_dim)
        
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.buffer_vertices = torch.tensor(args.buffer_vertices, dtype=torch.float32)
        self.num_vertices = self.buffer_vertices.shape[0]
        self.use_state_norm = args.use_state_norm
        
        # Orthogonal layer initialization
        orthogonal_init(self.layer_0)
        orthogonal_init(self.layer_1)
        orthogonal_init(self.layer_2, gain=0.01)
        
        ### Affine buffer
        self.buffer_center = torch.ones((self.num_vertices, 1)) @ self.buffer_vertices.mean(axis=0, keepdims=True)
        ### Initial buffer vertices of volume 0
        self.vertices = self.buffer_center.clone()
        
        ### Counting iterations to slowly enlarge the buffer once allowed
        self.iter = 0
        self.max_iter = args.max_iter_enlargment
        ### Initially not allowed to enlarge the buffer
        self.enlarge_buffer = False
     
    def enlarging_buffer(self):
        """Iteratively enlarges the affine buffer until reaching the desired volume."""
        self.iter += 1
        if self.iter > self.max_iter:
            self.enlarge_buffer = False
            print("Buffer reached full size")
        
        if self.iter % 10 == 0:
            a = self.iter/self.max_iter
            self.vertices = a*self.buffer_vertices + (1-a)*self.buffer_center
            

    def forward(self, s, state_norm=None, update=True):
        if self.enlarge_buffer:
            self.enlarging_buffer()

        if self.use_state_norm:
            s = state_norm(s, update=update)
            v = state_norm(self.vertices, update=False)
            with torch.no_grad():
                s = torch.cat( (s, v), dim=0)
        else:
            with torch.no_grad():
                s = torch.cat( (s, self.vertices), dim=0)
                
        s = torch.relu(self.layer_0(s, self.num_vertices))
        s = torch.relu(self.layer_1(s, self.num_vertices))
        s = self.layer_2(s, self.num_vertices)
        mean = s[:-self.num_vertices]
        return mean
            
    def get_dist(self, s, state_norm):
        mean = self.forward(s, state_norm)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


### Original without buffer enlargment
class POLICEd_Gaussian_Actor(nn.Module):
    def __init__(self, args):
        super(POLICEd_Gaussian_Actor, self).__init__()
        self.nb_layers = 3
        self.width = args.hidden_width
        self.layer_0 = ConstrainedLayer(args.state_dim, self.width)
        self.layer_1 = ConstrainedLayer(self.width, self.width)
        self.layer_2 = ConstrainedLayer(self.width, args.action_dim)
        
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.buffer_vertices = torch.tensor(args.buffer_vertices, dtype=torch.float32)
        self.num_vertices = self.buffer_vertices.shape[0]
        self.use_state_norm = args.use_state_norm
        
        # Orthogonal layer initialization
        orthogonal_init(self.layer_0)
        orthogonal_init(self.layer_1)
        orthogonal_init(self.layer_2, gain=0.01)

    def forward(self, s, state_norm=None, update=True):
        if self.use_state_norm:
            s = state_norm(s, update=update)
            v = state_norm(self.buffer_vertices, update=False)
            with torch.no_grad():
                s = torch.cat( (s, v), dim=0)
        else:
            with torch.no_grad():
                s = torch.cat( (s, self.buffer_vertices), dim=0)
                
        s = torch.relu(self.layer_0(s, self.num_vertices))
        s = torch.relu(self.layer_1(s, self.num_vertices))
        s = self.layer_2(s, self.num_vertices)
        mean = s[:-self.num_vertices]
        return mean
            
    def get_dist(self, s, state_norm):
        mean = self.forward(s, state_norm)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist



# Orthogonal initialization of NN layers
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


# unPOLICEd baseline Gaussian actor
class Gaussian_Actor(nn.Module):
    def __init__(self, args):
        super(Gaussian_Actor, self).__init__()
        self.nb_layers = 3
        self.width = args.hidden_width
        self.layer_0 = nn.Linear(args.state_dim, self.width)
        self.layer_1 = nn.Linear(self.width, self.width)
        self.layer_2 = nn.Linear(self.width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.use_state_norm = args.use_state_norm
        
        # Orthogonal initialization
        orthogonal_init(self.layer_0)
        orthogonal_init(self.layer_1)
        orthogonal_init(self.layer_2, gain=0.01)

    def forward(self, s, state_norm=None):
        if self.use_state_norm:
            s = state_norm(s)
        s = torch.relu(self.layer_0(s))
        s = torch.relu(self.layer_1(s))
        mean = self.layer_2(s)
        return mean

    def get_dist(self, s, state_norm):
        mean = self.forward(s, state_norm)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)

        # Orthogonal initialization
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        v_s = self.fc3(s)
        return v_s



class PPO():
    def __init__(self, args):
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_state_norm = args.use_state_norm

        if args.POLICEd:
            if args.enlarging_buffer:
                self.actor = Enlarging_POLICEd_Gaussian_Actor(args)
            else:
                self.actor = POLICEd_Gaussian_Actor(args)
        else:
            self.actor = Gaussian_Actor(args)
        self.critic = Critic(args)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s, state_norm=None):  # When evaluating the policy, we only use the mean
        if self.use_state_norm:
            assert state_norm is not None, "State normalization is required"
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s, state_norm).detach().numpy().flatten()
        return a

    def choose_action(self, s, state_norm=None):
        if self.use_state_norm:
            assert state_norm is not None, "State normalization is required"
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
       
        with torch.no_grad():
            dist = self.actor.get_dist(s, state_norm)
            a = dist.sample()  # Sample the action according to the probability distribution
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps, state_norm):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()  # Get training data
        batch_size = a.shape[0]
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            # for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index], state_norm)
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
            
       
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.optimizer_critic.state_dict(), filename + "_critic_optimizer")
		
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.optimizer_actor.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.optimizer_critic.load_state_dict(torch.load(filename + "_critic_optimizer"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.optimizer_actor.load_state_dict(torch.load(filename + "_actor_optimizer"))

