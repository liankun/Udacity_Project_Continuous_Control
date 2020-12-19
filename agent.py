from model import NetWorkActor, NetWorkCritic
from collections import namedtuple,deque
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
class OUNoise:
    """
    Ornstein-Uhlenbeck process
    """
    def __init__(self,size,seed,mu=0.,theta=0.15,sigma=0.2):
        """initialize"""
        self.mu=mu*np.ones(size)
        self.theta = theta
        self.sigma=sigma
        self.seed = random.seed(seed)
        self.state = copy.copy(self.mu)
    
    def sample(self):
        "generate noise"
        x = self.state
        dx = self.theta*(self.mu-x)+self.sigma*np.array([random.random() for i in range(len(x))])
        self.state = x+dx
        
        return self.state


class ReplayPool:
    """
    Fixed-size buffer to store experience tuples
    """
    def __init__(self,buffer_size,batch_size,device='cpu'):
        """
        buffer_size:maximum size of buffer
        batch_size: size of each trainning batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.device = device
    
    def add(self,state,action,reward,next_state,done):
        """
        add a new experience to memory
        """
        
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """
        randomly sample a batch of experience from memory
        """
        experiences = random.sample(self.memory,k=self.batch_size)
        
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        #print([e.reward.shape for e in experiences])
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        """
        return the current size of internal memory
        """
        
        return len(self.memory)
        

class Agent():
    """
    interact with the environment 
    """
    def __init__(self,state_size,
                 action_size,
                 replay_pool_size=int(1e5),
                 batch_size=128,
                 gamma=0.95,
                 update_step=4,
                 lr=5e-4,
                 tau=1e-3,
                 add_noise=False,
                 hidden_node1=64*2*2,
                 hidden_node2=32*2*2,
                 device='cpu'):
        """
        state_size:int, the size of state space
        action_size:int, the size of actoin space
        replay_pool_size: the size of replay size need to store
        batch_size: the size of minibatch used in learning
        gamma:discount rate
        tau: soft update rate
        update_step: how often to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.add_noise = add_noise
        
        self.noise = OUNoise(action_size,12345)
        
        #NetWork
        self.actor = NetWorkActor(state_size=state_size,action_size=action_size,
                          hidden_node1=hidden_node1,hidden_node2=hidden_node2).to(self.device)
        self.target_actor = NetWorkActor(state_size=state_size,action_size=action_size,
                               hidden_node1=hidden_node1,hidden_node2=hidden_node2).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        #the ouput size of the critic is 1
        #the input will combine the state and action
        self.critic = NetWorkCritic(state_size=state_size,action_size=action_size,
                           hidden_node1=hidden_node1,hidden_node2=hidden_node2).to(self.device)
        self.target_critic = NetWorkCritic(state_size=state_size,action_size=action_size,
                               hidden_node1=hidden_node1,hidden_node2=hidden_node2).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        #optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=lr)
        
        #create replay pool
        self.memory = ReplayPool(replay_pool_size,batch_size,device=self.device)
        
        self.t_step = 0
        self.update_step = update_step
        
    def step(self,state,action,reward,next_state,done):
        #put the experience into the pool
        self.memory.add(state,action,reward,next_state,done)
        
        #learn every update_step
        self.t_step = (self.t_step + 1)%self.update_step
        if self.t_step ==0:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def act(self,state,eps=0.):
        """
        return actions for given state as per current policy
        state : current state
        eps: scale for the noise
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state)
        self.actor.train()
        
        #add a normal noise scaled with eps
        action_values = action_values.cpu().data.numpy()
        if self.add_noise:
            noise = self.noise.sample()*eps
            action_values+=noise
        
        #all action values are between -1 and 1
        action_values = np.clip(action_values,-1,1)
      
        return action_values
        
    def learn(self,experiences):
        """
        update the qnetwork local
        experiences: tuple of (s,a,r,s',done) tuples
        gamma (float): discount factor
        """
        states,actions,rewards,next_states,dones = experiences
        
        #optimize critic 
        
        #print(rewards.shape)
        #print(q_next.shape)
        #print(dones.shape)
        
        self.critic_optimizer.zero_grad()
        q = self.critic.forward(states,actions)
        next_actions = self.target_actor.forward(next_states).detach()
        q_next = self.target_critic(next_states,next_actions)
        q_target = rewards+self.gamma*q_next*(1-dones)
        critic_loss = F.mse_loss(q_target,q)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #optimize actor
        self.actor_optimizer.zero_grad()
        pred_actions = self.actor(states)
        actor_loss = -self.critic.forward(states,pred_actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #soft update of target network
        self.soft_update()
        
    def soft_update(self):
        """
        weight_target = tau*weight_local+(1-tau)*weight_target
        """
        #soft update critic
        for target_param, param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        
        #soft update actor
        for target_param, param in zip(self.target_actor.parameters(),self.actor.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)