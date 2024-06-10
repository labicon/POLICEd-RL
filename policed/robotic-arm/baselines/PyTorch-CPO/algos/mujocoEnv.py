import mujoco as mj
import mujoco.viewer as vi
import numpy as np
import torch, copy, gym, time

#import os; os.add_dll_directory('C:\\Users\\kartik\\.mujoco\\mujoco200\\bin')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MujocoEnv(gym.Env):
    """Mujoco system environment."""
    
    def __init__(self, target = torch.tensor([0.5, 0.5, 0.5], device=device),
                       precision = 0.05,
                       max_magnitude = 0.1,
                       constraint_min = torch.tensor([0.41, 0.34, 0.57], device=device),
                       constraint_max = torch.tensor([0.59, 0.66, 0.57], device=device),
                       debug = False):

        ### Environment Parameters
        self.state_size = 10 # state dimension
        self.action_size = 7 # input dimension
        self.max_actuation = torch.tensor([2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05], device=device)
        self.action_min = -1*self.max_actuation
        self.action_max = 1*self.max_actuation
        self.last_action = torch.zeros(1, self.action_size)
        self.action_magnitude = max_magnitude

        # OpenAI definitions
        self.action_space = gym.spaces.Box(low=np.float32(self.action_min.cpu()), high=np.float32(self.action_max.cpu()), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_size,))
        self.dt = 0.1

        ### Mujoco Setup
        self.model = mj.MjModel.from_xml_path("../../robotic-arm-model/test_scene.xml")
        self.data = mj.MjData(self.model)
        self.viewer = vi.launch_passive(self.model, self.data)
        cam = self.viewer.cam
        cam.azimuth = 145.0 ; cam.elevation = -19.25 ; cam.distance = 2.0
        cam.lookat = np.array([ 0.2228265373149372 , 0.25857197832292195 , 0.7914213541765485 ])
        
        ### Target
        self.target = target
        self.precision = precision

        ### Forbidden to cross the line between constraint_min and constraint_max
        self.constraint_min = constraint_min
        self.constraint_max = constraint_max
        self.width_affine_area = 2*torch.tensor([0.09, 0.16, 0.05], device=device)

        self.constraint_max[2] += self.width_affine_area[2] # Switching the corner
        
        self.output_path = "./CPO_training_data_7.csv"
        with open(self.output_path, "a") as f:
            f.write("episode, total_reward, respect, done\n")
        self.counter = 0
        self.episode_stats = np.zeros((3,))

        self.site_index = -2

        self.debug = debug
        

    def getJoints(self):
        return torch.tensor(self.data.qpos, dtype=torch.float, device=device)
    
    def reward(self, state):
        return -torch.linalg.vector_norm(state - self.target)

    def getState(self):
        # Based on the current state definition
        return copy.deepcopy(torch.cat((self.state, self.last_action)))

    def crossing(self, new_state):
        new_state = torch.tensor(new_state, device=device)
        if((new_state > self.constraint_min).all() and (new_state < self.constraint_max).all()):
            return True
        if(torch.sign(new_state[2]-self.constraint_min[2]) != torch.sign(self.state[2]-self.constraint_min[2])):
            if(new_state[0] > self.constraint_min[0] and new_state[0] < self.constraint_max[0]):
                if(new_state[1] > self.constraint_min[1] and new_state[1] < self.constraint_max[1]):
                    return True
        return False

    def respect_constraint(self, new_state):
        """Checks whether new_state respects the constraint."""
        return not self.crossing(new_state)
    
    def arrived(self):
        """Episode is over when target is reached within given precision."""
        return torch.linalg.vector_norm(self.target - self.state) < self.precision
    
    def _propagation(self, u):
        """Common background function for 'step' and 'training_step'."""

        u = np.clip(u, a_min=-self.action_magnitude, a_max=self.action_magnitude)
        
        self.data.qpos += u
        self.last_action = torch.tensor(self.data.qpos, dtype=torch.float, device=device)
        mj.mj_step(self.model, self.data)
        # self.viewer.sync()
        # time.sleep(0.1)
        new_state = torch.tensor(self.data.site_xpos[self.site_index], dtype=torch.float, device=device)
        respect = self.respect_constraint(new_state)
        too_far = (self.last_action < self.action_min).any().item() or (self.last_action > self.action_max).any().item() or (new_state[2] < 0)

        return new_state, respect, too_far
            
    def step(self, u: np.ndarray): #  -> Tuple[torch.tensor, float, bool, bool]
        """Returns: new_state, reward, done, respect"""
        self.last_state = self.state
        new_state, respect, too_far = self._propagation(u)
        # self.state = new_state

        ### Update to new state only if constraint is not violated and hasn't exited allowed area
        if not too_far:
            self.state = new_state
        else:
            self._propagation(-u)
        
        done = self.arrived()
        reward = self.reward(self.state) + done# - 3*too_far - 6*(not respect)
        
        self.counter += 1
        self.episode_stats[0] += self.reward(self.state) + done - 3*too_far - 6*(not respect)
        self.episode_stats[1] += float(not respect)
        self.episode_stats[2] += float(done)

        return self.getState(), reward, done, respect

    def training_step(self, u):
        """Takes a step in the environment. If the constraint is violated or if
        the states exits the admissible area, the state is maintained in place
        and a penalty is assigned. This prevents too short trajectories.
        'done' is only assigned if the target is reached."""
        return self.step(u)
        

    def reset(self, max_dist_target=None): # base value should allow full state space sampling
        with open(self.output_path, "a") as f:
            f.write(f"{self.counter}, {self.episode_stats[0]}, {self.episode_stats[1]}, {self.episode_stats[2]}\n")
        self.counter = 0
        self.episode_stats = np.zeros((3,))
        
        """Resets the state of the linear system to a random state in [x_min, x_max]."""
        self.last_action = (self.action_max - self.action_min) * torch.rand(self.action_size, device=device) + self.action_min
        self.data.qpos = self.last_action.cpu()
        mj.mj_step(self.model, self.data)
        self.viewer.sync()
        self.state = torch.tensor(self.data.site_xpos[self.site_index], dtype=torch.float, device=device)
        self.last_state = self.state

        if max_dist_target is not None and torch.norm(self.state - self.target) > max_dist_target:
            self.state = self.reset(max_dist_target)

        return self.getState()
    
    def random_action(self):
        """Returns a uniformly random action within action space."""
        a = (self.action_max - self.action_min) * torch.rand(self.action_size) + self.action_min
        return torch.clamp(a, min=-self.action_magnitude, max=self.action_magnitude)

    def close(self):
        self.viewer.close()


# env = MujocoEnv()
# print(env)