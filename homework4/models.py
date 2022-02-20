from copy import deepcopy
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Callable, Tuple
from collections import namedtuple

from utils import polyak_average, as_tensor, set_seeds

import gym
from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward", "is_terminal"))


class DDPG(object): # off-policy algorithm
    def __init__(self, env: gym.Env, hidden_size: int, buffer_size: int, noise_std_max: float, noise_std_min: float,
                 gamma: float, tau: float, lr_a: float = 1e-3, lr_c: float = 1e-3, wd_a: float = 0, wd_c: float = 0):

        set_seeds(env, 42)
        self.env = env
        self.noise = OUNoise(self.env.action_space, max_sigma=noise_std_max, min_sigma=noise_std_min)
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Networks
        action_scale = max(env.action_space.high)
        self.actor = MLP(layers_sizes=[self.num_states, hidden_size, self.num_actions],
                           output_activation=nn.Tanh, action_scale=action_scale).to(self.device) # policy net ,
        self.actor_target = deepcopy(self.actor)
        self.critic = MLP([self.num_states + self.num_actions, hidden_size,
                           self.num_actions]).to(self.device) #action-value (Q) function
        self.critic_target = deepcopy(self.critic)

        for params in self.actor_target.parameters():
            params.requires_grad = False

        for params in self.critic_target.parameters():
            params.requires_grad = False
        
        # Training
        self.replay_buffer = ReplayBuffer(buffer_size)        
        self.critic_criterion = nn.MSELoss().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a, weight_decay=wd_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c, weight_decay=wd_c)
        #########################

    def get_action(self, state, step):
        """ 
        Calculate next action with exploration via Gaussian noise
        Remember to:
        1. Not calculate gradients for this forward pass.
        2. Include exploration noise.
        3. Clip action to fit within the environment's action space after adding the noise. 
        """
        #########################
        action = self.actor.forward(state).detach().cpu().numpy()
        action = self.noise.get_noised_action(action, step)
        #########################
        return action

    def critic_loss(self, batch):
        """
        Calculate critic's loss for a given batch.
        """
        #########################
        next_states = batch.next_state
        states = batch.state
        actions = batch.action
        rewards = batch.reward
        is_terminal = batch.is_terminal

        qvals = self.critic.forward(torch.cat([states, actions], 1))
        with torch.no_grad():
            next_actions = self.actor_target.forward(next_states)
            next_q = self.critic_target.forward(torch.cat([next_states, next_actions], 1))
        targets = rewards + self.gamma * (1 - is_terminal) * next_q
        critic_loss = self.critic_criterion(qvals, targets)
        #########################
        return critic_loss

    def actor_loss(self, batch):
        """
        Calculate actor's loss for a given batch.
        Remember that we want to maximize qvalue returned by critic, which represents the expected 
        future reward. Therefore, we want to do gradient ascent. However, optimizer always performs 
        gradient descent.
        """
        #########################
        states = batch.state
        actions = self.actor.forward(states)
        actor_loss = - self.critic.forward(torch.cat([states, actions], 1)).mean()
        #########################
        return actor_loss

    def update(self, batch_size):
        # Sample minibatch
        transitions = self.replay_buffer.sample(batch_size)
        # Convert from list of tuples to tuple of lists
        batch = Transition(*zip(*transitions))
        batch = Transition(as_tensor(batch.state, batch=True),
                           as_tensor(batch.action, batch=True),
                           as_tensor(batch.next_state, batch=True),
                           as_tensor(batch.reward, batch=True),
                           as_tensor(batch.is_terminal, batch=True))
        # Critic step ~ 4/5 lines
        #########################
        self.critic_optimizer.zero_grad()
        critic_loss = self.critic_loss(batch)
        critic_loss.backward() 
        self.critic_optimizer.step()
        #########################

        for params in self.critic.parameters():
            params.requires_grad = False
        # Actor step ~ 4/5lines
        #########################
        self.actor_optimizer.zero_grad()
        actor_loss = self.actor_loss(batch)
        actor_loss.backward()
        self.actor_optimizer.step()
        #########################

        for params in self.critic.parameters():
            params.requires_grad = True
        # Update target networks ~ 2 lines
        #########################
        with torch.no_grad():
            polyak_average(self.actor_target, self.actor, tau=self.tau)
            polyak_average(self.critic_target, self.critic, tau=self.tau)

        #########################
        return actor_loss, critic_loss

    def learn(self, batch_size: int, n_episodes: int, initial_exploration_steps: int, tensorboard_log_dir: str,
              checkpoint_save_interval: int, checkpoints_dir: str):
        total_steps, rewards_history = 0, []
        writer = SummaryWriter(tensorboard_log_dir)
        # writer.add_graph(self.actor, as_tensor(self.env.reset()).float())
        # writer.add_graph(self.critic, torch.cat([as_tensor(self.env.reset(), batch=True).float(),
        #                                          as_tensor(self.env.action_space.sample(), batch=True)], 1))

        for episode in range(n_episodes):
            # At the beginning explore randomly before using policy. This helps with exploration.
            episode_steps, episode_reward, done = 0, 0, False
            actor_loss, critic_loss = 0, 0
            state = self.env.reset()
            while not done:
                if total_steps < initial_exploration_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(as_tensor(state), total_steps)

                # Interaction with the environment, updates, logs.
                #########################
                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.add(Transition(state, action, next_state, reward, done))
                if len(self.replay_buffer) > batch_size:
                    actor_loss, critic_loss = self.update(batch_size)

                if (total_steps + 1) % checkpoint_save_interval == 0:
                    self.actor.save(f'{checkpoints_dir}/params_nsteps{total_steps + 1}_nepis{episode}')

                state = next_state

                # Misc
                total_steps += 1
                episode_steps += 1
                episode_reward += reward
                if actor_loss:
                    writer.add_scalar('Loss/MSBE_actor', actor_loss, total_steps)
                if actor_loss:
                    writer.add_scalar('Loss/MSBE_critic', critic_loss, total_steps)
                ######################### 

            rewards_history.append(episode_reward)
            # Tensorboard
            writer.add_scalar('Reward/episode', episode_reward, episode)
            writer.add_scalar('Reward/mean_50_episodes', np.mean(rewards_history[-50:]), episode)
            writer.add_scalar('Episode/n_steps', episode_steps, episode)
            writer.add_scalar('Episode/buffer_size', len(self.replay_buffer), episode)
            # writer.add_scalar('Misc/eps_exploration', self.exploration_fn._value, episode)
          
        writer.close()


class MLP(nn.Module):
    """ Simple MLP net.

    Each of the layers, despite the last one, is followed by `activation`, and the last one
    is optionally followed by `output_activation`.
    """
    def __init__(self, layers_sizes: List[int], activation: nn.Module = nn.ReLU, action_scale: float = 1.0,
                 output_activation: Optional[nn.Module] = False) -> None:
        super(MLP, self).__init__()
        self.action_scale = action_scale
        modules = []
        for in_features, out_features in zip(layers_sizes, layers_sizes[1:-1]):
            modules.extend([
                nn.Linear(in_features, out_features),
                activation()
            ])
        modules.extend([nn.Linear(layers_sizes[-2], layers_sizes[-1])])
        if output_activation:
            modules.extend([output_activation()])
        self.layers = nn.Sequential(*modules)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.layers(obs) * self.action_scale

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


class ReplayBuffer(object):
    def __init__(self, size: int):
        """ Create new replay buffer.

        Args:
            size: capacity of the buffer
        """
        self._storage = []
        self._capacity = size
        self._next_idx = 0

    def add(self, transition: Transition) -> None:
        if len(self._storage) < self._capacity:
            self._storage.append(None)
        self._storage[self._next_idx] = transition
        self._next_idx = (self._next_idx + 1) % self._capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """ Sample batch of experience from memory.

        Args:
            batch_size: size of the batch

        Returns:
            batch of transitions
        """
        batch = random.sample(self._storage, batch_size)
        return batch

    def __len__(self) -> int:
        return len(self._storage)
    
    
def gaussian_action_noise(action_dim: np.ndarray, mean: float, std: float) -> Callable[[], float]:
    """
    Returns function that samples noise of shape action_dim from the normal distribution N(mean, std) 
    """
    #########################
    return np.random.normal(mean, std)
    

class GaussianNoise(object):
    def __init__(self, action_space, mu=0.0, max_sigma=0.3, min_sigma=0.1, decay_period=100000):
        self.mu = mu
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high

    def noise(self):
        noise = self.mu + self.sigma * np.random.randn(self.action_dim)
        return noise

    def get_noised_action(self, action, step=0):
        ou_state = self.noise()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, step / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.1, decay_period=int(1e6)):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_noised_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)