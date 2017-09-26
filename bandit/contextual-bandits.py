# from this tutorial
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-1-5-contextual-bandits-bff01d1aad9c

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

learn_rate = 0.001
total_episode = 10000
e = 0.1


class ContextualBandit():
    def __init__(self):
        self.state = 0
        self.bandits = np.array([[0.2, 0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])
        self.num_bandits = self.bandits.shape[0]
        self.num_actions = self.bandits.shape[1]

    def get_bandit(self):
        self.state = np.random.randint(0, self.num_bandits)
        return self.state

    def pull_arm(self, action):
        bandit = self.bandits[self.state, action]
        result = np.random.randn()

        if result > bandit:
            return 1
        else:
            return -1


class Agent():
    def __init__(self, learn_rate, s_size, a_size):
        self.state = tf.placeholder(tf.int32, [1])
        state_onehot = slim.one_hot_encoding(self.state, s_size)
        output = slim.fully_connected(state_onehot, a_size, biases_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        # training procedure
        self.reward = tf.placeholder(tf.float32, [1])
        self.action = tf.placeholder(tf.int32, [1])
        self.weight = tf.slice(self.output, self.action,[1])
        self.loss = - tf.log(self.weight) * self.reward

        self.trainer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)


graph = tf.Graph()

with graph.as_default():

    ctx_bandit = ContextualBandit()
    agent = Agent(learn_rate,ctx_bandit.num_bandits, ctx_bandit.num_actions)

    total_reward = np.zeros([ctx_bandit.num_bandits, ctx_bandit.num_actions])
    weights = tf.trainable_variables()[0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for ep in range(total_episode):
            s = ctx_bandit.get_bandit()

            if np.random.rand() < e:
                act = np.random.randint(ctx_bandit.num_actions)
            else:
                act = sess.run(agent.chosen_action, feed_dict={agent.state: [s]})

            rew = ctx_bandit.pull_arm(act)

            total_reward[s, act] += rew

            w1, _ = sess.run([weights, agent.trainer], feed_dict={agent.reward:[rew], agent.action:[act], agent.state:[s]})

            if ep % 500 == 0:
                print(w1[s])
                print("Best action for bandit {} is {}".format(s, np.argmax(w1[s])))