# Simple Policy Gradient bandit on this tutorial
# https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149

import tensorflow as tf
import numpy as np

bandits = [0.2,0,-0.2,-5]
num_bandit = len(bandits)
total_episodes = 1000
total_reward = np.zeros(num_bandit)
e = 0.1

def pull_bandit(bandit):
    result = np.random.randn()

    if result > bandit:
        return 1
    else:
        return -1


graph = tf.Graph()
with graph.as_default():
    w = tf.Variable(tf.ones([num_bandit]))
    chose_action = tf.argmax(w,0)

    reward = tf.placeholder(tf.float32, [1])
    action = tf.placeholder(tf.int32,[1])
    w_a = tf.slice(w,action,[1])
    loss = - tf.log(w_a) * reward
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for ep in range(total_episodes):
            if np.random.rand() < e:
                act = np.random.randint(num_bandit)
            else:
                act = sess.run(chose_action)

            rew = pull_bandit(bandits[act])

            w1,_ = sess.run([w, trainer], feed_dict={reward:[rew], action:[act]})

            total_reward[act] += rew

            if ep % 50 ==0:
                print("Agent thinks {}th bandit is the most promising".format(np.argmax(w1)+1))
