import numpy as np


class QLearningAgent:
    def __init__(self, env, gamma=0.9, learning_rate=0.01, epsilon=1):
        self.gamma = gamma
        self.learing_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.nA
        self.state_n = env.nS
        self.q = np.zeros((env.nS, env.nA))

    def reset(self):
        self.q = np.zeros((self.state_n, self.action_n))

    def decide(self, state, factor):
        eps = self.epsilon * (1 - factor)
        if np.random.uniform() < eps:
            action = np.random.randint(self.action_n)
        else:
            action = self.q[state].argmax()
        return action
        # return 3

    def learn(self, state, action, reward, next_state):
        u = reward + self.gamma * self.q[next_state].max()
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learing_rate * td_error