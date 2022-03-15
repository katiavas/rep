import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Encoder(nn.Module):

    def __init__(self, input_dims, feature_dim=288):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        shape = self.conv_output(input_dims)
        self.fc1 = nn.Linear(shape, feature_dim)

    def conv_output(self, input_dims):
        state = T.zeros(1, *input_dims)
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.size())
        shape = x.size[0]*x.size[1]*x.size[2]*x.size[3]
        # return int(np.prod(x.size()))
        return shape

    def forward(self, state):
        conv = F.elu(self.conv1(state))
        conv = F.elu(self.conv2(conv))
        conv = F.elu(self.conv3(conv))
        conv = F.elu(self.conv4(conv))

        conv_flatten = conv.view((conv.size()[0], -1))
        features = self.fc1(conv_flatten)

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

    def forward(self, img, hx):
        
        state = self.encoder(img)
        
        hx = self.gru(state, hx)

        pi = self.pi(hx)
        v = self.v(hx)

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.numpy()[0], v, log_prob, hx

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

    def calc_cost(self, new_state, hx, done,
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


