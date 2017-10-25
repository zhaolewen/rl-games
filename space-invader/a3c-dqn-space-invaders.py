import random
import time
import scipy
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import multiprocessing
import threading
import scipy.signal as signal
import requests
import gym

def sendStatElastic(data, endpoint="http://35.187.182.237:9200/reinforce/games"):
    data['step_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        requests.post(endpoint, json=data)
    except:
        print("Elasticsearch exception")
        #log.warning(r.text)
    finally:
        pass

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

def make_gym_env(name):
    env = gym.make(name)
    return env

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

def discount_reward(rs, gamma):
    return signal.lfilter([1], [1, -gamma], rs[::-1], axis=0)[::-1]

def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_v,to_v in zip(from_vars, to_vars):
        op_holder.append(to_v.assign(from_v))

    return op_holder

def normalized_columns_initializer(std=1.0):
    def __initializer(shape, dtype=None, partition_info=None):
        out = np.random.rand(*shape).astype(np.float32)
        out *= std/np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return __initializer

class ACNetwork():
    def __init__(self, act_size, scope, trainer,frame_count=4,im_size=84, h_size=256, global_step=None):
        self.inputs = tf.placeholder(tf.float32, [None, im_size*im_size*frame_count], name="in_frames")
        img_in = tf.reshape(self.inputs, [-1, im_size, im_size, frame_count])

        conv1 = slim.convolution2d(scope="conv1",inputs=img_in, num_outputs=32, kernel_size=[8,8], stride=[4, 4], padding="VALID", biases_initializer=None)
        conv2 = slim.convolution2d(scope="conv2",inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding="VALID", biases_initializer=None)
        conv3 = slim.convolution2d(scope="conv3",inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding="VALID", biases_initializer=None)
        conv4 = slim.convolution2d(scope="conv4",inputs=conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding="VALID", biases_initializer=None)

        conv_flat = tf.reshape(slim.flatten(conv4), [-1, h_size])

        with tf.variable_scope("va_split"):
            stream_a, stream_v = tf.split(conv_flat,2,axis=1)
            w_a = tf.Variable(tf.random_normal([h_size//2, act_size]))
            w_v = tf.Variable(tf.random_normal([h_size//2, 1]))

            advantage = tf.matmul(stream_a, w_a)
            value = tf.matmul(stream_v, w_v)

        # salience = tf.gradients(advantage, img_in)
        with tf.variable_scope("predict"):
            self.q_out = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))
            self.pred = tf.argmax(self.q_out, axis=1)
            self.policy = tf.nn.softmax(advantage)


        # master network up date by copying value
        # workers by gradient descent
        if scope!="master":
            self.actions = tf.placeholder(tf.int32, [None],name="actions")
            act_onehot = tf.one_hot(self.actions, act_size, dtype=tf.float32)

            self.target_v = tf.placeholder(tf.float32, [None],name="target_v")
            self.target_adv = tf.placeholder(tf.float32, [None],name="target_advantage")

            resp_outputs = tf.reduce_sum(self.policy * act_onehot, [1])
            chosen_val = tf.reduce_sum(tf.multiply(self.q_out,act_onehot), axis=1)
            value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - chosen_val))

            entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
            policy_loss = - tf.reduce_sum(tf.log(resp_outputs) * self.target_adv)

            loss = 0.5 * value_loss + policy_loss - entropy * 0.01

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            gradients = tf.gradients(loss, local_vars)
            var_norms = tf.global_norm(local_vars)
            grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

            master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
            self.train_op = trainer.apply_gradients(zip(grads, master_vars), global_step=global_step)

            with tf.name_scope("summary"):
                tf.summary.scalar("loss", loss)
                tf.summary.scalar("mean_value", tf.reduce_mean(value))
                tf.summary.scalar("max_advantage", tf.reduce_max(advantage))
                tf.summary.scalar("min_advantage", tf.reduce_min(advantage))
                tf.summary.scalar("mean_target_q", tf.reduce_mean(self.target_v))
                tf.summary.scalar("mean_pred_q", tf.reduce_mean(self.q_out))

                self.summary_op = tf.summary.merge_all()


class Worker():
    def __init__(self, act_size , name, trainer, game_name,e_max=1.0,e_min=0.1,e_steps=100000,global_step=None,summary_writer=None):
        self.name = str(name)
        self.trainer = trainer
        self.act_size = act_size

        with tf.variable_scope(self.name):
            self.local_ac = ACNetwork(act_size, self.name, trainer, global_step=global_step)
        self.game_name = game_name
        self.e = e_max
        self.e_min=0.1
        self.e_delta = (e_max - e_min)/e_steps

        # copy values from master graph to local
        self.update_local_ops = update_target_graph('master', self.name)
        self.global_step = global_step
        self.summary_writer = summary_writer

    def train(self, rollout,gamma, bootstrap_val, sess):
        rollout = np.array(rollout)

        obs = rollout[:,0]
        acts = rollout[:,1]
        rewards = rollout[:,2]
        nxt_obs = rollout[:, 3]
        values = rollout[:, 5]

        reward_plus = np.asarray(rewards.tolist()+[bootstrap_val])
        disc_rew = discount_reward(reward_plus, gamma)[:-1]

        value_plus = np.asarray(values.tolist()+[bootstrap_val])
        #print(value_plus)
        advantages = rewards + gamma *value_plus[1:] - value_plus[:-1]
        advantages = discount_reward(advantages, gamma)

        feed_dict = {
            self.local_ac.inputs:np.vstack(obs),
            self.local_ac.target_v:disc_rew,
            self.local_ac.actions:acts,
            self.local_ac.target_adv:advantages,
        }

        _ ,step = sess.run([self.local_ac.train_op,self.global_step], feed_dict=feed_dict)
        #if self.summary_writer is not None:
        #    self.summary_writer.add_summary(summ,step)

    def play(self, sess, coord):
        print("Starting worker {}".format(self.name))
        env = gym.make(self.game_name)

        with sess.as_default():

            ep_count = 0
            while not coord.should_stop():
                frame_buffer = FrameBuffer(frame_size=84 * 84)

                s = env.reset()
                s = process_frame(s)
                frame_buffer.add(s)

                ep_score = 0.0
                t_ep_start = time.time()

                while True:
                    env.render()
                    pred = sess.run(self.local_ac.policy,feed_dict={self.local_ac.inputs: frame_buffer.frames()})

                    act = np.random.choice(range(self.act_size), p=pred[0])
                    s, reward, done, obs = env.step(act)
                    ep_score += reward

                    s = process_frame(s)
                    frame_buffer.add(s)

                    if done:
                        ep_count += 1
                        print("Agent {} finished episode {} finished with total reward: {} in {} seconds".format(self.name,ep_count, ep_score,
                                                                                               time.time() - t_ep_start))
                        sendStatElastic({"score": ep_score,'ageng_name':self.name, 'game_name': 'ac3-SpaceInvaders-v0', 'episode': ep_count})
                        break

    def work(self, gamma, sess, coord, max_ep_buffer_size=200, max_episode_count=10000):
        print("Starting worker {}".format(self.name))
        env = gym.make(self.game_name)


        with sess.as_default():

            ep_count = 0
            while not coord.should_stop() and ep_count<max_episode_count:
                sess.run(self.update_local_ops)

                frame_buffer = FrameBuffer(frame_size=84 * 84)

                s = env.reset()
                s = process_frame(s)
                frame_buffer.add(s)

                episode_buffer = []
                ep_score = 0.0
                t_ep_start = time.time()

                while True:
                    begin_frames = frame_buffer.frames()
                    pred, val = sess.run([self.local_ac.policy, self.local_ac.q_out],feed_dict={self.local_ac.inputs:begin_frames})
                    val = val[0]
                    act = np.random.choice(range(self.act_size),p=pred[0])
                    s, reward, done, obs = env.step(act)
                    ep_score += reward

                    s = process_frame(s)
                    reward = clip_reward_tan(reward)
                    frame_buffer.add(s)

                    next_frames = frame_buffer.frames()

                    episode_buffer.append([begin_frames, act, reward, next_frames, done, val[act]])

                    if len(episode_buffer) >= max_ep_buffer_size and not done:
                        pred,v_pred = sess.run([self.local_ac.pred, self.local_ac.q_out],
                                          feed_dict={self.local_ac.inputs:next_frames})
                        v_pred = v_pred[0]
                        self.train(episode_buffer, gamma,bootstrap_val=v_pred[pred], sess=sess)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if done:
                        ep_count += 1
                        print("Agent {} finished episode {} finished with total reward: {} in {} seconds".format(self.name,ep_count, ep_score, time.time()-t_ep_start))
                        sendStatElastic({"score": ep_score,'game_name': 'ac3-SpaceInvaders-v0','episode':ep_count})
                        break

                if len(episode_buffer) != 0:
                    self.train(episode_buffer, gamma, 0.0, sess)


if __name__=="__main__":
    game_name = 'SpaceInvaders-v0'

    logdir = "./checkpoints/a3c-dqn"

    max_episode_len = 10000
    action_count = 6
    gamma = 0.99
    #num_workers = multiprocessing.cpu_count() - 2
    num_workers = 6
    print("Running with {} workers".format(num_workers))

    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable("global_step",(),tf.int64,initializer=tf.zeros_initializer())
        trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

        #with tf.variable_scope("master"):
        master_worker = Worker(action_count,"master",trainer=None, game_name=game_name)
            #master_network = ACNetwork(act_size=action_count, scope="master", trainer=None)

        summ_writer = tf.summary.FileWriter(logdir)
        workers = []
        for k in range(num_workers):
            w_name = "worker_"+str(k)
            #with tf.variable_scope(w_name):
            w = Worker(action_count,w_name, trainer, game_name, summary_writer=summ_writer, global_step=global_step)
            workers.append(w)


    sv = tf.train.Supervisor(logdir=logdir, graph=graph, summary_op=None)

    with sv.managed_session() as sess:
        #with tf.Session() as sess:
        coord = tf.train.Coordinator()
        #sess.run(tf.global_variables_initializer())

        worker_threads = []
        for wk in workers:
            work = lambda : wk.work(gamma, sess, coord, max_episode_count=max_episode_len)
            t = threading.Thread(target=(work))

            t.start()

            time.sleep(0.5)
            worker_threads.append(t)

        master_worker.play(sess, coord)
        print("Started all threads")

        coord.join(worker_threads)
