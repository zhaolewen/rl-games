import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np
import scipy.misc
import time, requests

def sendStatElastic(data, endpoint="http://35.187.182.237:9200/reinforce/games"):
    data['step_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        requests.post(endpoint, json=data)
    except:
        print("Elasticsearch exception")
        #log.warning(r.text)
    finally:
        pass

def update_target_graph(from_scope, to_scope, tau):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_v,to_v in zip(from_vars, to_vars):
        op_holder.append(to_v.assign(from_v.value() * tau + (1-tau)*to_v.value()))

    return op_holder

class QNetwork():
    def __init__(self,h_size,action_size, reuse=None, img_size=84, learning_rate=0.0001):
        self.frame_in = tf.placeholder(tf.float32, [None, img_size * img_size * 3], name="frame_in")
        img_in = tf.reshape(self.frame_in, [-1,img_size, img_size, 3])

        conv1 = slim.convolution2d(scope="conv1",inputs=img_in, num_outputs=32, kernel_size=[8,8], stride=[4, 4], padding="VALID", biases_initializer=None)
        conv2 = slim.convolution2d(scope="conv2",inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding="VALID", biases_initializer=None)
        conv3 = slim.convolution2d(scope="conv3",inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding="VALID", biases_initializer=None)
        conv4 = slim.convolution2d(scope="conv4",inputs=conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding="VALID", biases_initializer=None)

        self.train_len = tf.placeholder(tf.int32,[])
        self.batch_size = tf.placeholder(tf.int32, [])
        conv_flat = tf.reshape(slim.flatten(conv4), [self.batch_size, self.train_len, h_size])

        cell = tf.nn.rnn_cell.BasicLSTMCell(h_size, state_is_tuple=True, reuse=False)
        self.state_init = cell.zero_state(self.batch_size, tf.float32)
        rnn, self.rnn_state = tf.nn.dynamic_rnn(cell, conv_flat,dtype=tf.float32, initial_state=self.state_init)
        #print(rnn)
        #print(self.rnn_state)
        rnn = tf.reshape(rnn, [-1, h_size])

        with tf.variable_scope("va_split", reuse=reuse):
            stream_a, stream_v = tf.split(rnn,2,axis=1)
            w_a = tf.Variable(tf.random_normal([h_size//2, action_size]))
            w_v = tf.Variable(tf.random_normal([h_size//2, 1]))

            advantage = tf.matmul(stream_a, w_a)
            value = tf.matmul(stream_v, w_v)

        # salience = tf.gradients(advantage, img_in)
        with tf.variable_scope("predict", reuse=reuse):
            self.q_out = value + tf.subtract(advantage, tf.reduce_mean(advantage, 1, keep_dims=True))
            self.pred = tf.argmax(self.q_out, axis=1)

            self.target_q = tf.placeholder(tf.float32, [None])
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions, action_size,dtype=tf.float32)

            Q = tf.reduce_sum(tf.multiply(self.q_out, actions_onehot), axis=1)

            td_error = tf.square(self.target_q - Q)

            loss = tf.reduce_mean(td_error)

        self.update = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)

    def predict_act(self, frame_list, init_state,trace_len=1, batch_size=1, session=None):
        dict = {
            self.frame_in: frame_list,
            self.train_len: trace_len,
            self.state_init: init_state,
            self.batch_size: batch_size
        }
        act, q_vals, rnn_s = session.run([self.pred, self.q_out, self.rnn_state], feed_dict=dict)

        return act, q_vals,rnn_s

    def update_nn(self, in_frame, target_q_val, acts, trace_len, init_state, batch_size, session):
        update_dict = {
            self.frame_in: in_frame,
            self.target_q: target_q_val,
            self.actions: acts,
            self.train_len: trace_len,
            self.state_init: init_state,
            self.batch_size: batch_size
        }
        session.run(self.update, feed_dict=update_dict)


class ExperienceBuffer():
    def __init__(self, buffer_size=10000):
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

def process_frame(f, height=84,width=84):
    f = scipy.misc.imresize(f, (height, width))
    return np.reshape(f,[-1])/255.0

def discounted_reward(rs, gamma):
    total = 0
    for k in reversed(range(len(rs))):
        total = total * gamma + rs[k]

    return total

if __name__=="__main__":
    game_name = 'SpaceInvaders-v0'
    env = gym.make(game_name)

    batch_size = 4 # num of experience traces
    trace_len = 16
    update_step = 10

    gamma = 0.999 # discount factor for reward
    e_start = 1.0 # prob of random action
    e_end = 0.1
    annel_steps  = 40000.0 # steps from e_start to e_end
    total_episodes = 10000

    pre_train_steps = 10000 # steps of random action before training begins
    logdir = "./checkpoints"

    h_size = 512
    tau = 0.001
    action_size = env.action_space.n

    e_delta = (e_start - e_end) / annel_steps
    exp_buffer = ExperienceBuffer()

    scope_main = "main_qn"
    scope_target = "target_qn"

    # build graph
    graph = tf.Graph()
    with graph.as_default():
        with tf.variable_scope(scope_main):
            main_qn = QNetwork(h_size, action_size)
        with tf.variable_scope(scope_target):
            target_qn = QNetwork(h_size, action_size)

    sv = tf.train.Supervisor(logdir=logdir, graph=graph)
    e = e_start
    total_step = 0

    with sv.managed_session() as sess:
        update_qn_op = update_target_graph(scope_main, scope_target, tau)

        for ep in range(total_episodes):
            ep_buffer = []
            s = env.reset()
            s_frame = process_frame(s)

            rnn_state = (np.zeros([1, h_size]),np.zeros([1, h_size]))
            ep_rewards = []
            last_act = 0

            while True:
                env.render()
                # normal process
                act,_, state_rnn1 = main_qn.predict_act([s_frame], rnn_state, session=sess)
                act = act[0]
                if np.random.rand() < e or total_step<pre_train_steps:
                    act = np.random.randint(0, action_size)

                last_act = act

                s1, reward, done, obs = env.step(act)
                s1_frame = process_frame(s1)

                ep_buffer.append(np.reshape(np.array([s_frame, act, reward, s1_frame, done]), [1,5]))

                if total_step > pre_train_steps:
                    if e > e_end:
                        e -= e_delta

                    if total_step % update_step == 0:
                        sess.run(update_qn_op)
                        # update model
                        state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
                        train_batch = exp_buffer.sample(batch_size, trace_len)

                        pred_act, _, _ = main_qn.predict_act(np.vstack(train_batch[:, 3]),state_train, trace_len, batch_size, sess)
                        _, q_vals, _ = target_qn.predict_act(np.vstack(train_batch[:, 3]),state_train, trace_len, batch_size, sess)

                        end_multiplier = - (train_batch[:, 4] - 1)
                        double_q = q_vals[range(batch_size * trace_len),pred_act]
                        target_q_val = train_batch[:, 2] + gamma * double_q * end_multiplier

                        in_frames = np.vstack(train_batch[:, 0])
                        acts = train_batch[:,1]
                        main_qn.update_nn(in_frames, target_q_val, acts, trace_len, state_train, batch_size, sess)

                s = s1
                s_frame = s1_frame
                rnn_state = state_rnn1

                ep_rewards.append(reward)
                total_step += 1

                if done:
                    disc_r = discounted_reward(ep_rewards, gamma)
                    score = discounted_reward(ep_rewards, 1)

                    print("Episode {} finished with discounted reward {}, score {}, e {}".format(ep, disc_r, score,e))
                    sendStatElastic({"discount_reward":disc_r, "score":score,"episode":ep,"rand_e_prob":e,'game_name':game_name})
                    break

            # add episode to experience buffer
            step_buffer = np.array(ep_buffer)
            step_buffer = list(zip(step_buffer))
            exp_buffer.add(step_buffer)