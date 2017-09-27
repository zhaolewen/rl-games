import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np

env = gym.make('MsPacman-v0')

total_episodes = 100

class QNetwork():
    def __init__(self,h_size,action_size, in_size=21168, img_size=84, learning_rate=0.0001):
        self.frame_in = tf.placeholder(tf.float32, [None, in_size])
        img_in = tf.reshape(self.frame_in, [-1,img_size, img_size, 3])
        conv1 = slim.convolution2d(inputs=img_in, num_outputs=32, kernel_size=[8,8], stride=[4, 4], padding="VALID", biases_initializer=None)
        conv2 = slim.convolution2d(inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding="VALID", biases_initializer=None)
        conv3 = slim.convolution2d(inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding="VALID", biases_initializer=None)
        conv4 = slim.convolution2d(inputs=conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding="VALID", biases_initializer=None)

        self.train_len = tf.placeholder(tf.int32)
        self.batch_size = tf.placeholder(tf.int32)
        conv_flat = tf.reshape(slim.flatten(conv4), [self.batch_size, self.train_len, h_size])

        cell = tf.nn.rnn_cell.LSTMCell(10)
        rnn = tf.nn.dynamic_rnn(cell, conv_flat,dtype=tf.float32)
        rnn = tf.reshape(rnn, [-1, h_size])

        stream_a, stream_v = tf.split(rnn,2,1)
        w_a = tf.Variable(tf.random_normal([h_size//2, 4]))
        w_v = tf.Variable(tf.random_normal([h_size//2, 1]))

        advantage = tf.matmul(stream_a, w_a)
        value = tf.matmul(stream_v, w_v)

        salience = tf.gradients(advantage, img_in)
        self.q_out = value + tf.subtract(advantage, tf.reduce_mean(advantage, 1, keep_dims=True))
        self.pred = tf.argmax(self.q_out)

        self.target_q = tf.placeholder(tf.float32, [None])
        self.actions = tf.placeholder(tf.int32, [None])
        actions_onehot = tf.one_hot(self.actions, action_size,dtype=tf.float32)

        Q = tf.reduce_sum(tf.multiply(self.q_out, actions_onehot), axis=1)

        td_error = tf.square(self.target_q - Q)

        mask_a = tf.zeros([self.batch_size, self.train_len//2])
        mask_b = tf.ones([self.batch_size, self.train_len//2])

        mask = tf.concat([mask_a, mask_b], 1)
        mask = tf.reshape(mask, [-1])

        self.loss = tf.reduce_mean(td_error * mask)

        self.update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


class ExperienceBuffer():
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size, trace_len):
        samples_ep = random.sample(self.buffer, batch_size)
        samples_traces = []
        for episode in samples_ep:
            pt = np.random.randint(0, len(episode) + 1 - trace_len)
            samples_traces.append(episode[pt:pt+trace_len])

        samples_traces = np.array(samples_traces)
        return np.reshape(samples_traces, [batch_size * trace_len, 5]) # why 5 ?


for ep in range(total_episodes):
    env.reset()

    while True:
        env.render()
        action = env.action_space.sample()

        state, reward, done, obs = env.step(action)

        if done:
            print("Episode {} finished".format(ep))
            break