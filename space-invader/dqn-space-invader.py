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

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_v,to_v in zip(from_vars, to_vars):
        op_holder.append(tf.assign(to_v, from_v))

    return op_holder

class QNetwork():
    def __init__(self,h_size,action_size, img_size=84, learning_rate=0.00025):
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

        with tf.variable_scope("va_split"):
            stream_a, stream_v = tf.split(rnn,2,axis=1)
            w_a = tf.Variable(tf.random_normal([h_size//2, action_size]))
            w_v = tf.Variable(tf.random_normal([h_size//2, 1]))

            advantage = tf.matmul(stream_a, w_a)
            value = tf.matmul(stream_v, w_v)

        # salience = tf.gradients(advantage, img_in)
        with tf.variable_scope("predict"):
            self.q_out = value + tf.subtract(advantage, tf.reduce_mean(advantage, 1, keep_dims=True))
            self.pred = tf.argmax(self.q_out, axis=1)

            self.target_q = tf.placeholder(tf.float32, [None])
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions, action_size,dtype=tf.float32)

            Q = tf.reduce_sum(tf.multiply(self.q_out, actions_onehot), axis=1)

            td_error = tf.square(self.target_q - Q)

            loss = tf.reduce_mean(td_error)

        #self.update = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
        self.update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.name_scope("summary"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("mean_value", tf.reduce_mean(value))
            tf.summary.scalar("max_advantage", tf.reduce_max(advantage))
            tf.summary.scalar("mean_target_q", tf.reduce_mean(self.target_q))
            tf.summary.scalar("mean_pred_q", tf.reduce_mean(self.q_out))

            self.summary_op = tf.summary.merge_all()

    def predict_act(self, frame_list, init_state,trace_len=1, batch_size=1, session=None):
        dict = {
            self.frame_in: frame_list,
            self.train_len: trace_len,
            self.state_init: init_state,
            self.batch_size: batch_size
        }
        act, q_vals, rnn_s = session.run([self.pred, self.q_out, self.rnn_state], feed_dict=dict)

        return act, q_vals,rnn_s

    def update_nn(self, in_frame, target_q_val, acts, trace_len, init_state, batch_size, session, writer=None, step=None):
        update_dict = {
            self.frame_in: in_frame,
            self.target_q: target_q_val,
            self.actions: acts,
            self.train_len: trace_len,
            self.state_init: init_state,
            self.batch_size: batch_size
        }
        _, summ = session.run([self.update, self.summary_op], feed_dict=update_dict)
        if writer is not None and step is not None:
            writer.add_summary(summ, step)


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
        return np.reshape(samples_traces, [batch_size * trace_len, 5])

def process_frame(f, height=84,width=84, im_file=None):
    f = scipy.misc.imresize(f, (height, width))
    if im_file is not None:
        scipy.misc.imsave(im_file, f)
    return np.reshape(f,[-1])/255.0

def discounted_reward(rs, gamma):
    total = 0
    for k in reversed(range(len(rs))):
        total = total * gamma + rs[k]

    return total

if __name__=="__main__":
    game_name = 'SpaceInvaders-v0'
    env = gym.make(game_name)

    batch_size = 32 # num of experience traces
    trace_len = 6
    update_step = 5
    update_target_step = 10000

    gamma = 0.99 # discount factor for reward
    e_start = 1.0 # prob of random action
    e_end = 0.1
    annel_steps  = 100000 # steps from e_start to e_end
    total_episodes = 10000

    pre_train_steps = 30000 # steps of random action before training begins
    logdir = "./checkpoints"
    pic_dir = "./pics/"

    h_size = 512
    action_size = env.action_space.n
    skip_frame = 4
    save_frame_step = 5000

    e_delta = (e_start - e_end) / annel_steps
    exp_buffer = ExperienceBuffer()

    scope_main = "main_qn"
    scope_target = "target_qn"

    # build graph
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable("global_step",(),tf.int64,tf.zeros_initializer(), trainable=False)
        inc_global_step = tf.assign(global_step, global_step.value()+1)
        summ_writer = tf.summary.FileWriter(logdir)

        with tf.variable_scope(scope_main):
            main_qn = QNetwork(h_size, action_size)
        with tf.variable_scope(scope_target):
            target_qn = QNetwork(h_size, action_size)

    sv = tf.train.Supervisor(logdir=logdir, graph=graph, summary_op=None)
    e = e_start
    total_step = 0

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with sv.managed_session(config=config) as sess:
        update_qn_op = update_target_graph(scope_main, scope_target)

        for ep in range(total_episodes):
            ep_buffer = []
            s = env.reset()
            s_frame = process_frame(s)

            rnn_state = (np.zeros([1, h_size]),np.zeros([1, h_size]))
            ep_rewards = []
            last_act = 0
            t_ep_start = time.time()

            while True:
                env.render()
                if total_step%skip_frame ==0:
                    s1, reward, done, obs = env.step(last_act)
                else:
                    # normal process
                    act,_, state_rnn1 = main_qn.predict_act([s_frame], rnn_state, session=sess)
                    act = act[0]
                    if np.random.rand() < e or total_step<pre_train_steps:
                        act = np.random.randint(0, action_size)

                    last_act = act

                    s1, reward, done, obs = env.step(act)
                    if total_step % save_frame_step == 0:
                        s1_frame = process_frame(s1, im_file=pic_dir+str(total_step)+".jpg")
                    else:
                        s1_frame = process_frame(s1)

                    ep_buffer.append(np.reshape(np.array([s_frame, act, reward, s1_frame, done]), [1,5]))

                    if total_step > pre_train_steps:
                        if e > e_end:
                            e -= e_delta

                        if total_step % update_step == 0:
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
                            main_qn.update_nn(in_frames, target_q_val, acts, trace_len, state_train, batch_size, sess, summ_writer, step_value)

                    s = s1
                    s_frame = s1_frame
                    rnn_state = state_rnn1

                ep_rewards.append(reward)
                total_step += 1

                if total_step % update_target_step == 0:
                    _, step_value = sess.run([update_qn_op, inc_global_step])
                else:
                    step_value = sess.run(inc_global_step)

                if done:
                    disc_r = discounted_reward(ep_rewards, gamma)
                    score = discounted_reward(ep_rewards, 1)

                    print("Episode {} finished in {} seconds with discounted reward {}, score {}, e {}, global step {}".format(ep, time.time()-t_ep_start, disc_r, score,e, step_value))
                    sendStatElastic({"discount_reward":disc_r, "score":score,"episode":ep,"rand_e_prob":e,'game_name':game_name})
                    break

            # add episode to experience buffer
            step_buffer = np.array(ep_buffer)
            step_buffer = list(zip(step_buffer))
            exp_buffer.add(step_buffer)