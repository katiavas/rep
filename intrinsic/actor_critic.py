import pickle

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os


class Encoder(nn.Module):

    def __init__(self, input_dims, feature_dim=288):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, (3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)

        shape = self.get_conv_out(input_dims)
        # Layer that will extract the features
        self.fc1 = nn.Linear(shape, feature_dim)

    def get_conv_out(self, input_dims):
        img = T.zeros(1, *input_dims)
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        shape = x.size()[0]*x.size()[1]*x.size()[2]*x.size()[3]
        # return int(np.prod(x.size()))
        return shape

    def forward(self, img):
        enc = F.elu(self.conv1(img))
        enc = F.elu(self.conv2(enc))
        enc = F.elu(self.conv3(enc))
        enc = F.elu(self.conv4(enc))

        enc_flatten = T.flatten(enc, start_dim=1)
        # enc_flatten = enc.view((enc.size()[0], -1))
        features = self.fc1(enc_flatten)

        return features


class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99, tau=1.0, feature_dims=288):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.tau = tau
        self.encoder = Encoder(input_dims)

        self.gru = nn.GRUCell(feature_dims, 256)
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)
        # self.checkpoint_file = os.path.join('intrinsic/', 'actor')
        # self.actor_critic = ActorCritic(input_dims=input_dims, n_actions=n_actions)

    # It will take a state/image and a hidden state for our GRU as an input
    # def forward(self, state, hx):
    def forward(self, img, hx):

        state = self.encoder(img)
        hx = self.gru(state, hx)

        # Pass hidden state into our pi and v layer to get our logs for our policy(pi) and out value function
        pi = self.pi(hx)
        v = self.v(hx)

        # Choose action function/ Get the actual probability distribution
        probs = T.softmax(pi, dim=1) # soft max activation on the first dimension
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # return predicted action, value, log probability and hidden state
        return action.numpy()[0], v, log_prob, hx

    def save_models(self, input_dims, n_actions):
        '''with open(os.path.join('./', 'fn.pickle'), 'wb') as handle:
            pickle.dump(self.actor_critic, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
        np.save(os.path.join('./', 'actor'), ActorCritic(input_dims=input_dims, n_actions=n_actions))
        print('... saving models ...')

    def load_models(self):
        np.load(os.path.join('./', 'actor'))



    def calc_R(self, done, rewards, values):
        values = T.cat(values).squeeze()

        if len(values.size()) == 1:  # batch of states
            R = values[-1]*(1-int(done))
        elif len(values.size()) == 0:  # single state
            R = values*(1-int(done))

        batch_return = []
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return,
                                dtype=T.float).reshape(values.size())
        return batch_return

    def calc_loss(self, new_state, hx, done,
                  rewards, values, log_probs, intrinsic_reward=None):

        if intrinsic_reward is not None:
            rewards += intrinsic_reward.detach().numpy()

        returns = self.calc_R(done, rewards, values)

        next_v = T.zeros(1, 1) if done else self.forward(T.tensor(
                                        [new_state], dtype=T.float), hx)[1]
        values.append(next_v.detach())
        values = T.cat(values).squeeze()
        log_probs = T.cat(log_probs)
        rewards = T.tensor(rewards)

        delta_t = rewards + self.gamma * values[1:] - values[:-1]
        n_steps = len(delta_t)
        gae = np.zeros(n_steps)
        for t in range(n_steps):
            for k in range(0, n_steps-t):
                temp = (self.gamma*self.tau)**k * delta_t[t+k]
                gae[t] += temp
        gae = T.tensor(gae, dtype=T.float)

        actor_loss = -(log_probs * gae).sum()
        # if single then values is rank 1 and returns rank 0
        # want to have same shape to avoid a warning
        critic_loss = F.mse_loss(values[:-1].squeeze(), returns)

        entropy_loss = (-log_probs * T.exp(log_probs)).sum()

        total_loss = actor_loss + critic_loss - 0.01 * entropy_loss
        return total_loss


