# from this tutorial
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

import gym
import numpy as np

env = gym.make('FrozenLake-v0')

Q = np.zeros([env.observation_space.n, env.action_space.n])

learn_rate = 0.8
gamma = 0.95

num_episode = 10000


for ep in range(num_episode):
    s = env.reset()

    r_sum = 0
    while True:
        #env.render()
        a = np.argmax(Q[s,:]+np.random.randn(1, env.action_space.n)*(1./(ep+1)))

        state, reward, done, obs = env.step(a)
        r_sum += reward

        Q[s,a] += learn_rate * (reward + gamma * np.max(Q[state,:])-Q[s,a])
        s = state

        if done:
            print("Episode {} finished with total reward {}".format(ep, reward))
            break