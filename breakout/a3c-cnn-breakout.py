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
import tensorflow.contrib.layers as layers

from tensorflow.python.training import training_ops
from tensorflow.python.training import slot_creator

class RMSPropApplier(object):
    def __init__(self,
               learning_rate,
               decay=0.9,
               momentum=0.0,
               epsilon=1e-10,
               clip_norm=40.0,
               device="/cpu:0",
               name="RMSPropApplier"):

        self._name = name
        self._learning_rate = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._epsilon = epsilon
        self._clip_norm = clip_norm
        self._device = device

        # Tensors for learning rate and momentum.  Created in _prepare.
        self._learning_rate_tensor = None
        self._decay_tensor = None
        self._momentum_tensor = None
        self._epsilon_tensor = None

        self._slots = {}

    def _create_slots(self, var_list):
        for v in var_list:
              # 'val' is Variable's intial value tensor.
              val = tf.constant(1.0, dtype=v.dtype, shape=v.get_shape())
              self._get_or_make_slot(v, val, "rms", self._name)
              self._zeros_slot(v, "momentum", self._name)

    def _prepare(self):
          self._learning_rate_tensor = tf.convert_to_tensor(self._learning_rate,
                                                          name="learning_rate")
          self._decay_tensor = tf.convert_to_tensor(self._decay, name="decay")
          self._momentum_tensor = tf.convert_to_tensor(self._momentum,
                                                     name="momentum")
          self._epsilon_tensor = tf.convert_to_tensor(self._epsilon,
                                                    name="epsilon")

    def _slot_dict(self, slot_name):
        named_slots = self._slots.get(slot_name, None)
        if named_slots is None:
              named_slots = {}
              self._slots[slot_name] = named_slots
        return named_slots

    def _get_or_make_slot(self, var, val, slot_name, op_name):
        named_slots = self._slot_dict(slot_name)
        if var not in named_slots:
            named_slots[var] = slot_creator.create_slot(var, val, op_name)
        return named_slots[var]

    def get_slot(self, var, name):
        named_slots = self._slots.get(name, None)
        if not named_slots:
            return None
        return named_slots.get(var, None)

    def _zeros_slot(self, var, slot_name, op_name):
        named_slots = self._slot_dict(slot_name)
        if var not in named_slots:
            named_slots[var] = slot_creator.create_zeros_slot(var, op_name)
        return named_slots[var]

    # TODO: in RMSProp native code, memcpy() (for CPU) and
    # cudaMemcpyAsync() (for GPU) are used when updating values,
    # and values might tend to be overwritten with results from other threads.
    # (Need to check the learning performance with replacing it)
    def _apply_dense(self, grad, var):
        rms = self.get_slot(var, "rms")
        mom = self.get_slot(var, "momentum")
        return training_ops.apply_rms_prop(
          var, rms, mom,
          self._learning_rate_tensor,
          self._decay_tensor,
          self._momentum_tensor,
          self._epsilon_tensor,
          grad,
          use_locking=False).op

    # Apply accumulated gradients to var.
    def apply_gradients(self, var_list, accum_grad_list, name=None):
        update_ops = []

        with tf.device(self._device):
            with tf.control_dependencies(None):
                self._create_slots(var_list)

        with tf.name_scope(name, self._name, []) as name:
            self._prepare()
            for var, accum_grad in zip(var_list, accum_grad_list):
                with tf.name_scope("update_" + var.op.name), tf.device(var.device):
                    clipped_accum_grad = tf.clip_by_norm(accum_grad, self._clip_norm)
                    update_ops.append(self._apply_dense(clipped_accum_grad, var))
            return tf.group(*update_ops, name=name)

def sendStatElastic(data, endpoint="http://35.187.182.237:9200/reinforce/games"):
    data['step_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        r = requests.post(endpoint, json=data)
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
    env.env.frameskip=3
    return env

def process_frame(f, last_f=None, height=84,width=84):
    if last_f is not None:
        f = np.amax(np.array([f, last_f]), axis=0)

    f = scipy.misc.imresize(f[34:194,:160,:], (height, width))
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

def exp_coeff(vs, gamma):
    for k in range(len(vs)):
        vs[k] *= gamma ** (k+1)

    return vs

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
    def __init__(self, act_size, scope, grad_applier=None, init_learn_rate=1e-3, learn_rate_decay_step=1e8,frame_count=4,im_size=84, h_size=256, global_step=None):
        self.inputs = tf.placeholder(tf.float32, [None, im_size*im_size*frame_count], name="in_frames")
        img_in = tf.reshape(self.inputs, [-1, im_size, im_size, frame_count])

        #conv1 = slim.convolution2d(activation_fn=tf.nn.relu,scope="conv1",inputs=img_in, num_outputs=32, kernel_size=[8,8], stride=[4, 4], padding="VALID", biases_initializer=None)
        #conv2 = slim.convolution2d(activation_fn=tf.nn.relu,scope="conv2",inputs=conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding="VALID", biases_initializer=None)
        #conv3 = slim.convolution2d(activation_fn=tf.nn.relu,scope="conv3",inputs=conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding="VALID", biases_initializer=None)
        #conv4 = slim.convolution2d(activation_fn=tf.nn.relu,scope="conv4",inputs=conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding="VALID", biases_initializer=None)

        with tf.name_scope("conv"):
            # conv1 = layers.conv2d(img_in, num_outputs=32, kernel_size=[5,5], stride=1, padding="VALID", weights_initializer=layers.xavier_initializer())
            # pool1 = layers.max_pool2d(conv1, kernel_size=[2,2], stride=2)
            # conv2 = layers.conv2d(pool1, num_outputs=32, kernel_size=[5,5], stride=1, padding="VALID", weights_initializer=layers.xavier_initializer())
            # pool2 = layers.max_pool2d(conv2, kernel_size=[2,2], stride=2)
            # conv3 = layers.conv2d(pool2, num_outputs=64, kernel_size=[4,4], stride=1, padding="VALID", weights_initializer=layers.xavier_initializer())
            # pool3 = layers.max_pool2d(conv3, kernel_size=[2,2], stride=2)
            # conv4 = layers.conv2d(pool3, num_outputs=64, kernel_size=3, stride=1, padding="VALID", weights_initializer=layers.xavier_initializer())
            # pool4 = layers.max_pool2d(conv4, kernel_size=[2,2], stride=2)
            # hidden = layers.fully_connected(layers.flatten(pool4), h_size, weights_initializer=layers.xavier_initializer())

            conv1 = layers.conv2d(img_in, num_outputs=16, kernel_size=[8, 8], stride=[4, 4])
            conv2 = layers.conv2d(conv1, num_outputs=32, kernel_size=[4, 4], stride=[2, 2])
            hidden = layers.fully_connected(layers.flatten(conv2), h_size)
        #hidden = slim.flatten(conv4)

        with tf.variable_scope("va_split"):
            advantage = slim.fully_connected(hidden, act_size, activation_fn=None, weights_initializer=normalized_columns_initializer(std=0.01))
            self.value = slim.fully_connected(hidden, 1, activation_fn=None, weights_initializer=normalized_columns_initializer(std=1.0))

        # salience = tf.gradients(advantage, img_in)
        with tf.variable_scope("predict"):
            #self.q_out = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))
            self.pred = tf.argmax(advantage, axis=1)
            self.policy = tf.nn.softmax(advantage)
            #self.policy = tf.clip_by_value(self.policy, 1e-13,1.0)


        # master network up date by copying value
        # workers by gradient descent
        if scope!="master":
            self.actions = tf.placeholder(tf.int32, [None],name="actions")
            act_onehot = tf.one_hot(self.actions, act_size, dtype=tf.float32)

            self.target_v = tf.placeholder(tf.float32, [None],name="target_v")
            self.target_adv = tf.placeholder(tf.float32, [None],name="target_advantage")
            #self.entropy_scale = tf.placeholder(tf.float32,[],name="entrypy_scale")

            resp_outputs = tf.reduce_sum(self.policy * act_onehot, [1])
            value_loss = tf.reduce_mean(tf.square(self.target_v - self.value))

            entropy = -tf.reduce_mean(tf.reduce_sum(self.policy * tf.log(self.policy+1e-13), axis=1))
            policy_loss = - tf.reduce_mean(tf.log(resp_outputs+1e-13) * self.target_adv)

            loss = 0.5 * value_loss + policy_loss - entropy * 0.001

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            gradients = tf.gradients(loss, local_vars)
            #var_norms = tf.global_norm(local_vars)
            grads, grad_norms = tf.clip_by_global_norm(gradients, 5.0)

            master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
            learning_rate = tf.Variable(init_learn_rate, trainable=False,dtype=tf.float32, name="learning_rate")
            delta_learn_rate = init_learn_rate/learn_rate_decay_step
            self.decay_learn_rate = tf.assign(learning_rate,learning_rate.value() - delta_learn_rate)

            #trainer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.0, decay=0.99, epsilon=1e-6)
            self.train_op = grad_applier.apply_gradients(master_vars, grads)
            self.incr_global_step = global_step.assign(global_step.value()+1)

            with tf.name_scope("summary"):
                s_lr = tf.summary.scalar("learning_rate", learning_rate)
                s_loss = tf.summary.scalar("loss", loss)
                s_val = tf.summary.scalar("mean_value", tf.reduce_mean(self.value))
                s_max_adv = tf.summary.scalar("max_advantage", tf.reduce_max(advantage))
                s_min_adv = tf.summary.scalar("min_advantage", tf.reduce_min(advantage))
                s_tar_q = tf.summary.scalar("mean_target_q", tf.reduce_mean(self.target_v))
                s_v_l = tf.summary.scalar("value_loss", value_loss)
                s_p_l = tf.summary.scalar("policy_loss", policy_loss)
                s_en = tf.summary.scalar("entropy", entropy)
                #s_pred_q = tf.summary.scalar("mean_pred_q", tf.reduce_mean(self.q_out))

                self.summary_op = tf.summary.merge([ s_lr,s_loss, s_val, s_max_adv, s_min_adv, s_tar_q, s_v_l,s_p_l,s_en])

def get_exp_prob(step, max_step=500000):
    min_p = np.random.choice([0.1,0.01,0.5],1,p=[0.4,0.3,0.3])[0]
    #min_p = 0.1
    if step > max_step:
        return min_p
    return 1.0 - (1.0 - min_p)/max_step*step

class Worker():
    def __init__(self, act_size , name, grad_applier=None, game_name=None,global_step=None,summary_writer=None):
        self.name = str(name)
        #self.trainer = trainer
        self.act_size = act_size

        with tf.variable_scope(self.name):
            self.local_ac = ACNetwork(act_size, self.name, grad_applier, global_step=global_step)
        self.game_name = game_name

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

        if bootstrap_val is None or np.isnan(bootstrap_val)==True:
            bootstrap_val = 0

        reward_plus = np.asarray(rewards.tolist()+[bootstrap_val])
        disc_rew = discount_reward(reward_plus, gamma)[:-1]

        value_plus = np.asarray(values.tolist()+[bootstrap_val])
        #print(value_plus)
        #advantages = disc_rew + exp_coeff(value_plus[1:], gamma) - value_plus[:-1]
        advantages = disc_rew + gamma * value_plus[1:] - value_plus[:-1]
        #advantages = discount_reward(advantages, gamma)
        #advantages = disc_rew - values

        feed_dict = {
            self.local_ac.inputs:np.vstack(obs),
            self.local_ac.target_v:disc_rew,
            self.local_ac.actions:acts,
            self.local_ac.target_adv:advantages,
        }

        summ,_ ,step,_ = sess.run([self.local_ac.summary_op, self.local_ac.train_op,self.local_ac.incr_global_step, self.local_ac.decay_learn_rate], feed_dict=feed_dict)
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summ,step)

    def play(self, sess, coord, render=False):
        print("Starting worker {}".format(self.name))
        env = make_gym_env(self.game_name)
        total_step = 0

        with sess.as_default():

            ep_count = 0
            while not coord.should_stop():
                frame_buffer = FrameBuffer(frame_size=84 * 84)

                s = env.reset()
                s = process_frame(s)
                frame_buffer.add(s)

                ep_score = 0.0
                t_ep_start = time.time()

                ep_len = 0
                while True:
                    total_step += 1
                    ep_len += 1
                    if render:
                        env.render()
                    pred = sess.run(self.local_ac.policy,feed_dict={self.local_ac.inputs: frame_buffer.frames()})

                    act = np.random.choice(range(self.act_size), p=pred[0])
                    #act = pred[0]
                    s, reward, done, obs = env.step(act)
                    ep_score += reward

                    s = process_frame(s)
                    frame_buffer.add(s)

                    if done:
                        ep_count += 1
                        print("Agent {} finished episode {} finished with total reward: {} in {} seconds, total step {}".format(self.name,ep_count, ep_score,
                                                                                               time.time() - t_ep_start,total_step))
                        sendStatElastic({"score": ep_score,'agent_name':self.name, 'game_name': 'ac3-Breakout-v0', 'episode': ep_count,'frame_count':total_step,'episode_length':ep_len})
                        break

    def work(self, gamma, sess, coord, max_ep_buffer_size=8, max_episode_count=5000):
        print("Starting worker {}".format(self.name))
        env = make_gym_env(self.game_name)
        total_step = 0

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

                e = get_exp_prob(total_step)
                ep_len = 0
                while True:
                    total_step += 1
                    ep_len += 1

                    begin_frames = frame_buffer.frames()
                    pred, val = sess.run([self.local_ac.policy, self.local_ac.value],feed_dict={self.local_ac.inputs:begin_frames})
                    val = val[0,0]
                    #e = get_exp_prob(total_step)
                    #if random.random() < e:
                    #    act = np.random.choice(range(self.act_size))
                    #else:
                    act = np.random.choice(range(self.act_size), p=pred[0])
                        #act = pred[0]
                    s, reward, done, obs = env.step(act)
                    ep_score += reward

                    s = process_frame(s)
                    reward = clip_reward(reward)
                    frame_buffer.add(s)

                    next_frames = frame_buffer.frames()

                    episode_buffer.append([begin_frames, act, reward, next_frames, done, val])

                    if len(episode_buffer) >= max_ep_buffer_size and not done:
                        v_pred = sess.run(self.local_ac.value,feed_dict={self.local_ac.inputs:next_frames})
                        self.train(episode_buffer, gamma,bootstrap_val=v_pred[0,0], sess=sess)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if done:
                        ep_count += 1
                        print("Agent {} finished episode {} finished with total reward: {} in {} seconds, total step {}".format(self.name,ep_count, ep_score, time.time()-t_ep_start, total_step))
                        sendStatElastic({"score": ep_score,'game_name': 'ac3-Breakout-v0','episode':ep_count,'rand_e_prob':100.0*e,'agent_name':self.name,'frame_count':total_step,'episode_length':ep_len})
                        break

                if len(episode_buffer) != 0:
                    self.train(episode_buffer, gamma, 0.0, sess)


if __name__=="__main__":
    game_name = 'Breakout-v0'

    logdir = "./checkpoints/a3c-dqn"

    max_episode_len = 10000
    action_count = 4
    gamma = 0.99
    #num_workers = multiprocessing.cpu_count() - 2
    num_workers = 32
    train_step = 8
    print("Running with {} workers".format(num_workers))

    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.get_variable("global_step",(),tf.int64,initializer=tf.zeros_initializer(), trainable=False)
        #learning_rate = tf.Variable(0.00025, trainable=False, dtype=tf.float32, name="learning_rate")
        #trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
        #trainer = tf.train.RMSPropOptimizer(learning_rate=0.00025, momentum=0.0, decay=0.99, epsilon=1e-6)
        #trainer = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.95)
        #trainer = tf.train.AdadeltaOptimizer(learning_rate=1e-4)
        grad_applier = RMSPropApplier(learning_rate=0.00025)

        #with tf.variable_scope("master"):
        master_worker = Worker(action_count,"master", game_name=game_name)
            #master_network = ACNetwork(act_size=action_count, scope="master", trainer=None)

        summ_writer = tf.summary.FileWriter(logdir)
        workers = []
        for k in range(num_workers):
            w_name = "worker_"+str(k)
            #with tf.variable_scope(w_name):
            w = Worker(action_count,w_name, grad_applier, game_name, summary_writer=summ_writer, global_step=global_step)
            workers.append(w)


    sv = tf.train.Supervisor(logdir=logdir, graph=graph, summary_op=None)

    with sv.managed_session() as sess:
        #with tf.Session() as sess:
        coord = tf.train.Coordinator()
        #sess.run(tf.global_variables_initializer())

        worker_threads = []
        for wk in workers:
            work = lambda : wk.work(gamma, sess, coord, max_episode_count=max_episode_len, max_ep_buffer_size=train_step)
            t = threading.Thread(target=(work))

            t.start()

            time.sleep(0.5)
            worker_threads.append(t)

        #master_worker.play(sess, coord, render=True)
        print("Started all threads")

        coord.join(worker_threads)
