
import numpy as np

import torch
import torch.nn as nn

from model import (Actor, Critic)

class DDPG(object):
    def __init__(self, states_num, actions_num, args):

        self.states_num = states_num
        self.actions_num = actions_num
        self.epsilon = 1.0

        self.actor = Actor(self.states_num, self.actions_num, args.hidden1, args.hidden2)
        self.actor_target = Actor(self.states_num, self.actions_num, args.hidden1, args.hidden2)

        self.critic = Critic(self.states_num, self.actions_num, args.hidden1, args.hidden2)
        self.critic_target = Critic(self.states_num, self.actions_num, args.hidden1, args.hidden2)

        initialize(self.actor_target, self.actor) # Make sure target is with the same weight
        initialize(self.critic_target, self.critic)
        
        #Create replay buffer
        '''
        self.memory
        '''

        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.epsilon_decrease_factor = 1.0 / args.epsilon

        self.last_state = None
        self.last_action = None

    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Target q batch
        next_q_values = self.critic_target([to_tensor(next_state_batch, volatile=True), self.actor_target(to_tensor(next_state_batch, volatile=True))])
        next_q_values.volatile=False

        target_q_batch = to_tensor(reward_batch) + self.discount * to_tensor(terminal_batch.astype(np.float)) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch), to_tensor(action_batch) ])
        
        value_loss = nn.MSELoss(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([to_tensor(state_batch), self.actor(to_tensor(state_batch))])

        policy_loss = policy_loss.mean()
        policy_loss.backward()

        # Target update
        update(self.actor_target, self.actor, self.tau)
        update(self.critic_target, self.critic, self.tau)

    def observe(self, reward_now, state_next, done):
        self.memory.append(self.last_state, self.last_action, reward_now, done)
        self.last_state = state_next

    def random_action(self):
        action = np.random.uniform(-100.,100.,self.actions_num)
        self.last_action = action
        return action

def initialize(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )