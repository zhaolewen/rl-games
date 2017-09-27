# from the tutorial
# https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-3-model-based-rl-9a6fe0cce99

import numpy as np
import gym
import tensorflow as tf

env = gym.make('CartPole-v0')

# policy network params
n_hid = 8
learn_rate = 0.01
gamma = 0.99
decay_rate = 0.99
in_dim = 4

# model network params
m_hid = 256


def resetGradBuffer(gradBuffer):
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


# This function uses our model to produce a new state when given a previous state and action
def stepModel(sess, xs, action, pred_state, prev_state):
    toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
    myPredict = sess.run([pred_state], feed_dict={prev_state: toFeed})
    reward = myPredict[0][:, 4]
    observation = myPredict[0][:, 0:4]
    observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
    observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
    doneP = np.clip(myPredict[0][:, 5], 0, 1)
    if doneP > 0.1 or len(xs) >= 300:
        done = True
    else:
        done = False
    return observation, reward, done

graph = tf.Graph()
with graph.as_default():
    # policy network
    observations = tf.placeholder(tf.float32,[None, 4], name="observations")
    W1 = tf.get_variable("W1", shape=[4, n_hid], initializer=tf.contrib.layers.xavier_initializer())
    layer1 = tf.nn.relu(W1)

    W2 = tf.get_variable("W2", shape=[n_hid, 1], initializer=tf.contrib.layers.xavier_initializer())
    score = tf.matmul(layer1, W2)
    prob = tf.nn.sigmoid(score)

    tvars = tf.trainable_variables()
    input_y = tf.placeholder(tf.float32,[None, 1], name="input_y")
    advantges = tf.placeholder(tf.float32, name="reward")
    adam = tf.train.AdamOptimizer(learning_rate=learn_rate)
    W1Grad = tf.placeholder(tf.float32, name="batch_grad1")
    W2Grad = tf.placeholder(tf.float32, name="batch_grad2")

    batch_grad = [W1Grad, W2Grad]
    loglik = tf.log(input_y * (input_y - prob) + (1-input_y)*(input_y + prob))
    loss = tf.reduce_mean(loglik * advantges)
    new_grads = tf.gradients(loss, tvars)

    update_grads = adam.apply_gradients(zip(batch_grad, tvars))

    # model network
    input_data = tf.placeholder(tf.float32, [None, 5])
    with tf.variable_scope("rnnlm"):
        soft_w = tf.get_variable("softmax_w", [m_hid, 50])
        soft_b = tf.get_variable("softmax_b", [50])


    prev_state = tf.placeholder(tf.float32, [None, 5], name="previous_state")
    w1m = tf.get_variable("w1m", shape=[5, m_hid], initializer=tf.contrib.layers.xavier_initializer())
    b1m = tf.get_variable("b1m", shape=[m_hid], initializer=tf.zeros_initializer())
    layer1m = tf.nn.relu(tf.matmul(prev_state, w1m) + b1m)

    w2m = tf.get_variable("w2m", shape=[m_hid, m_hid], initializer=tf.contrib.layers.xavier_initializer())
    b2m = tf.get_variable("b2m", shape=[m_hid], initializer=tf.zeros_initializer())
    layer2m = tf.nn.relu(tf.matmul(layer1m, w2m)+b2m)

    wo = tf.get_variable("wo", shape=[m_hid, 4], initializer=tf.contrib.layers.xavier_initializer())
    wr = tf.get_variable("wr", shape=[m_hid, 1], initializer=tf.contrib.layers.xavier_initializer())
    wd = tf.get_variable("wd", shape=[m_hid, 1], initializer=tf.contrib.layers.xavier_initializer())

    bo = tf.Variable(tf.zeros([4]), name="bo")
    br = tf.Variable(tf.zeros([1]), name="br")
    bd = tf.Variable(tf.ones([1]), name="bd")

    pred_obs = tf.matmul(layer2m, wo, name="pred_observation") + bo
    pred_reward = tf.matmul(layer2m, wr, name="pred_reward") + br
    pred_done = tf.sigmoid(tf.matmul(layer2m, wd, name="pred_done") + bd)

    true_obs = tf.placeholder(tf.float32, [None, 4], name="true_observation")
    true_reward = tf.placeholder(tf.float32, [None, 1], name="true_reward")
    true_done = tf.placeholder(tf.float32, [None, 1], name="true_done")

    pred_state = tf.concat([pred_obs, pred_reward, pred_done], 1)
    obs_loss = tf.square(true_obs - pred_obs)
    reward_loss = tf.square(true_reward - pred_reward)
    done_loss = tf.mul(pred_done, true_done) + tf.mul(1-pred_done, 1-true_done)
    done_loss = -tf.log(done_loss)

    model_loss = tf.reduce_mean(obs_loss + done_loss + reward_loss)
    m_adam = tf.train.AdamOptimizer(learning_rate=learn_rate)
    m_update = m_adam.minimize(model_loss)
