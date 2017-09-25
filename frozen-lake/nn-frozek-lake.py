# from this tutorial
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
# the Q-network approach

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
learn_rate = 0.8
gamma = 0.99
e = 0.1

num_episode = 2000

graph = tf.Graph()
with graph.as_default():
    in_state = tf.placeholder(tf.float32, [1,16])
    W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
    Q_out = tf.matmul(in_state, W)

    pred = tf.argmax(Q_out, 1)

    next_Q = tf.placeholder(tf.float32, [1,4])
    loss = tf.reduce_sum(tf.square(next_Q - Q_out))

    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    r_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for ep in range(num_episode):
            s = env.reset()

            r_sum = 0
            while True:
                # env.render()
                a, q_vals = sess.run([pred, Q_out], feed_dict={in_state:np.identity(16)[s:s+1]})

                if np.random.rand() < e:
                    a[0] = env.action_space.sample()

                state, reward, done, obs = env.step(a[0])

                q_pred = sess.run(Q_out, feed_dict={in_state:np.identity(16)[state:state+1]})

                max_q_pred = np.max(q_pred)

                target_q = q_vals
                target_q[0,a[0]] = reward + gamma * max_q_pred

                sess.run(trainer, feed_dict={in_state:np.identity(16)[s:s+1], next_Q:target_q})
                r_sum += reward

                s = state

                if done:
                    print("Episode {} finished with total reward {}".format(ep, r_sum))
                    r_list.append(r_sum)
                    e = 1./(ep/50+10)
                    break

plt.plot(r_list)
plt.savefig("nn_frozen_lake_reward.png")
plt.close()



