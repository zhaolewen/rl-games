import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np
import scipy.misc
import time, requests
import PIL
from tensorflow.core.framework import summary_pb2

def sendStatElastic(data, endpoint="http://35.187.182.237:9200/reinforce/games"):
    data['step_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        requests.post(endpoint, json=data)
    except:
        print("Elasticsearch exception")
        #log.warning(r.text)
    finally:
        pass

def update_target_graph(from_scope, to_scope, tau=0.001):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_v,to_v in zip(from_vars, to_vars):
        op_holder.append(tf.assign(to_v, from_v.value() * tau + to_v.value() * (1.0-tau)))

    return op_holder

class QNetwork():
    def __init__(self,h_size,action_size, img_size=84, learning_rate=0.00025, frame_count=4):
        self.frame_in = tf.placeholder(tf.float32, [None, img_size * img_size * frame_count], name="frame_in")
        img_in = tf.reshape(self.frame_in, [-1,img_size, img_size, frame_count])

        conv1 = slim.convolution2d(scope="conv1",inputs=img_in, num_outputs=32, kernel_size=[8,8], stride=[4, 4], padding="VALID", biases_initializer=None)
        conv2 = slim.convolution2d(scope="conv2",inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding="VALID", biases_initializer=None)
        conv3 = slim.convolution2d(scope="conv3",inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding="VALID", biases_initializer=None)
        conv4 = slim.convolution2d(scope="conv4",inputs=conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding="VALID", biases_initializer=None)

        self.batch_size = tf.placeholder(tf.int32, [])
        conv_flat = tf.reshape(slim.flatten(conv4), [self.batch_size, h_size])

        with tf.variable_scope("va_split"):
            stream_a, stream_v = tf.split(conv_flat,2,axis=1)
            w_a = tf.Variable(tf.random_normal([h_size//2, action_size]))
            w_v = tf.Variable(tf.random_normal([h_size//2, 1]))

            advantage = tf.matmul(stream_a, w_a)
            value = tf.matmul(stream_v, w_v)

        # salience = tf.gradients(advantage, img_in)
        with tf.variable_scope("predict"):
            self.q_out = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))
            self.pred = tf.argmax(self.q_out, axis=1)

            self.target_q = tf.placeholder(tf.float32, [None])
            self.actions = tf.placeholder(tf.int32, [None])
            actions_onehot = tf.one_hot(self.actions, action_size,dtype=tf.float32)

            Q = tf.reduce_sum(tf.multiply(self.q_out, actions_onehot), axis=1)

            #td_error = tf.square(self.target_q - Q)
            #loss = tf.reduce_mean(td_error)
            loss = tf.losses.huber_loss(self.target_q,Q)

        self.update = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.95).minimize(loss)

        with tf.name_scope("summary"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("mean_value", tf.reduce_mean(value))
            tf.summary.scalar("max_advantage", tf.reduce_max(advantage))
            tf.summary.scalar("mean_target_q", tf.reduce_mean(self.target_q))
            tf.summary.scalar("mean_pred_q", tf.reduce_mean(self.q_out))

            self.summary_op = tf.summary.merge_all()

    def predict_act(self, frame_list, batch_size=1, session=None):
        dict = {
            self.frame_in: frame_list,
            self.batch_size: batch_size
        }
        act, q_vals = session.run([self.pred, self.q_out], feed_dict=dict)

        return act, q_vals

    def update_nn(self, in_frame, target_q_val, acts, batch_size, session, writer=None, step=None):
        update_dict = {
            self.frame_in: in_frame,
            self.target_q: target_q_val,
            self.actions: acts,
            self.batch_size: batch_size
        }
        _, summ = session.run([self.update, self.summary_op], feed_dict=update_dict)
        if writer is not None and step is not None:
            writer.add_summary(summ, step)


class ExperienceBuffer():
    def __init__(self, buffer_size=100000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        samples_ep = random.sample(self.buffer, batch_size)

        return np.reshape(np.array(samples_ep), [batch_size, 5])


class FrameBuffer():
    def __init__(self, buffer_size=4, frame_size=None):
        self.buffer_size = buffer_size
        self.frame_size = frame_size
        self._frames = [[0] * frame_size] * buffer_size

    def add(self, frame):
        self._frames.append(frame)
        if len(self._frames) > self.buffer_size:
            self._frames.pop(0)

    def frames(self):
        return np.reshape(np.array(self._frames), [1, self.frame_size * self.buffer_size])


def process_frame(f, last_f=None, height=84,width=84):
    if last_f is not None:
        f = np.amax(np.array([f, last_f]), axis=0)
    f = scipy.misc.imresize(f, (height, width))
    f = np.dot(f[...,:3], [0.299, 0.587, 0.114])/255.0

    return np.reshape(f,[-1])

def clip_reward(r):
    if r>0:
        return 1.0
    elif r<0:
        return -1.0

    return 0

def clip_reward_tan(r):
    return np.arctan(r)

def discounted_reward(rs, gamma):
    total = 0
    for k in reversed(range(len(rs))):
        total = total * gamma + rs[k]

    return total

if __name__=="__main__":
    game_name = 'SpaceInvaders-v0'
    env = gym.make(game_name)
    game_name += '-ddqn-cnn'
    env.frameskip = 4

    render = False

    batch_size = 32 # num of experience traces
    update_target_step = 10000

    gamma = 0.99 # discount factor for reward
    e_start = 1.0 # prob of random action
    e_end = 0.1
    annel_steps  = 1000000 # steps from e_start to e_end
    total_episodes = 90000
    update_step = 4
    tau = 0.001
    exp_buffer_size = 100000

    pre_train_steps = 20000 # steps of random action before training begins
    logdir = "./checkpoints/ddqn-cnn"

    h_size = 512
    action_size = env.action_space.n
    frame_count = 4
    img_size = 84

    e_delta = (e_start - e_end) / annel_steps
    exp_buffer = ExperienceBuffer(buffer_size=exp_buffer_size)

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

        #update_qn_op = update_target_graph(scope_main, scope_target, tau)
        copy_graph_op = update_target_graph(scope_main, scope_target, 1.0)

    sv = tf.train.Supervisor(logdir=logdir, graph=graph, summary_op=None)
    e = e_start
    total_step = 0

    with sv.managed_session() as sess:
        step_value,_ = sess.run([global_step, copy_graph_op])

        for ep in range(total_episodes):
            frame_buffer = FrameBuffer(buffer_size=frame_count, frame_size=img_size*img_size)

            s = env.reset()
            s_frame = process_frame(s)
            frame_buffer.add(s_frame)

            ep_rewards = []
            last_act = 0
            t_ep_start = time.time()

            last_frame = None

            while True:
                if render:
                    env.render()

                begin_frames = frame_buffer.frames()

                if np.random.rand() < e or total_step<pre_train_steps:
                    act = np.random.randint(0, action_size)
                else:
                    act, _ = main_qn.predict_act(begin_frames, session=sess)
                    act = act[0]

                last_act = act

                s1, reward, done, _ = env.step(act)

                r2 = clip_reward_tan(reward)
                s1_frame = process_frame(s1, last_frame)
                last_frame = s1

                frame_buffer.add(s1_frame)
                next_frames = frame_buffer.frames()
                exp_buffer.add(np.reshape(np.array([begin_frames, act, r2, next_frames, done]), [1,5]))

                if total_step > pre_train_steps:
                    if e > e_end:
                        e -= e_delta

                    if total_step % update_step == 0:
                        # update model
                        train_batch = exp_buffer.sample(batch_size)

                        pred_act, _ = main_qn.predict_act(np.vstack(train_batch[:, 3]), batch_size, sess)
                        _, q_vals = target_qn.predict_act(np.vstack(train_batch[:, 3]), batch_size, sess)

                        end_multiplier = - (train_batch[:, 4] - 1)
                        double_q = q_vals[range(batch_size),pred_act]
                        target_q_val = train_batch[:, 2] + gamma * double_q * end_multiplier

                        in_frames = np.vstack(train_batch[:, 0])
                        acts = train_batch[:,1]
                        main_qn.update_nn(in_frames, target_q_val, acts, batch_size, sess, summ_writer, step_value)
                        step_value = sess.run(inc_global_step)

                        # register rand prob
                        summary = tf.Summary()
                        summary.value.add(tag='rand_prob', simple_value=e)
                        summ_writer.add_summary(summary, step_value)
                        summ_writer.flush()

                    s = s1
                    s_frame = s1_frame

                ep_rewards.append(reward)
                total_step += 1

                if total_step % update_target_step == 0:
                    sess.run(copy_graph_op)

                if done:
                    disc_r = discounted_reward(ep_rewards, gamma)
                    score = discounted_reward(ep_rewards, 1)

                    print("Episode {} finished in {} seconds with discounted reward {}, score {}, e {}, global step {}".format(ep, time.time()-t_ep_start, disc_r, score,e, step_value))
                    sendStatElastic({"discount_reward":disc_r, "score":score,"episode":ep,"rand_e_prob":e,'game_name':game_name})
                    break