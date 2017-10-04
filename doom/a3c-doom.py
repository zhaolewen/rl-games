from vizdoom import *
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

def sendStatElastic(data, endpoint="http://35.187.182.237:9200/reinforce/games"):
    data['step_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        requests.post(endpoint, json=data)
    except:
        print("Elasticsearch exception")
        #log.warning(r.text)
    finally:
        pass

def get_doom_game():
    game = DoomGame()
    game.set_doom_scenario_path("basic.wad")
    game.set_doom_map("map01")

    # either config in the file, or config in the code
    #game.load_config("vizdoom/scenarios/basic.config")
    game.add_available_button(Button.ATTACK)
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    #game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.add_available_game_variable(GameVariable.AMMO2)
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.set_episode_timeout(500)
    game.set_episode_start_time(10)
    game.set_living_reward(-1)
    game.set_mode(Mode.PLAYER)
    game.set_sound_enabled(False)

    game.init()

    return game

def process_frame(fr):
    s = fr[10:-10, 30:-30]
    #print(np.shape(s))
    s = scipy.misc.imresize(s, [84, 84])
    s = np.reshape(s, [-1])/255.0

    return s

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
    def __init__(self, im_size, act_size, scope, trainer, h_size=256):
        self.inputs = tf.placeholder(tf.float32, [None, im_size])
        img_in = tf.reshape(self.inputs, [-1, 84, 84, 1])

        conv1 = slim.conv2d(img_in, num_outputs=16, kernel_size=[8,8], stride=[4,4], padding="VALID", activation_fn=tf.nn.elu)
        conv2 = slim.conv2d(conv1, num_outputs=32,kernel_size=[4,4], stride=[2,2], padding="VALID", activation_fn=tf.nn.elu)
        hidden = slim.fully_connected(slim.flatten(conv2), h_size, activation_fn=tf.nn.elu)
        rnn_in = tf.expand_dims(hidden, 0)

        # batch size as sequence length
        step_size = tf.shape(img_in)[:1]

        cell = tf.nn.rnn_cell.BasicLSTMCell(h_size, state_is_tuple=True)

        self.state_init = cell.zero_state(1, tf.float32)

        rnn_out, rnn_state  = tf.nn.dynamic_rnn(cell, rnn_in, initial_state=self.state_init, sequence_length=step_size)
        rnn_c, rnn_h = rnn_state
        self.state_out = (rnn_c[:1,:], rnn_h[:1,:])
        rnn_out = tf.reshape(rnn_out, [-1, h_size])

        self.policy = slim.fully_connected(rnn_out, act_size, activation_fn=tf.nn.softmax, biases_initializer=None, weights_initializer=normalized_columns_initializer(0.01))
        self.value = slim.fully_connected(rnn_out, 1, activation_fn=None, biases_initializer=None, weights_initializer=normalized_columns_initializer(1.0))

        # master network up date by copying value
        # workers by gradient descent
        if scope!="master":
            self.actions = tf.placeholder(tf.int32, [None])
            act_onehot = tf.one_hot(self.actions, act_size, dtype=tf.float32)

            self.target_v = tf.placeholder(tf.float32, [None])
            self.advantages = tf.placeholder(tf.float32, [None])

            resp_outputs = tf.reduce_sum(self.policy * act_onehot, [1])
            value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
            entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))
            policy_loss = - tf.reduce_sum(tf.log(resp_outputs) * self.advantages)

            loss = 0.5 * value_loss + policy_loss - entropy * 0.01

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            gradients = tf.gradients(loss, local_vars)
            var_norms = tf.global_norm(local_vars)
            grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

            master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'master')
            self.grad_apply = trainer.apply_gradients(zip(grads, master_vars))


class Worker():
    def __init__(self,in_size, act_size , name, trainer, game):
        self.name = "worker_"+str(name)
        self.trainer = trainer
        self.act_size = act_size

        with tf.variable_scope(self.name):
            self.local_ac = ACNetwork(in_size, act_size, self.name, trainer)
        self.env = game
        self.actions = np.identity(act_size, dtype=bool).tolist()

        # copy values from master graph to local
        self.update_local_ops = update_target_graph('master', self.name)

    def train(self, rollout,gamma, bootstrap_val, sess):
        rollout = np.array(rollout)

        obs = rollout[:,0]
        acts = rollout[:,1]
        rewards = rollout[:,2]
        nxt_obs = rollout[:, 3]
        values = rollout[:, 5]

        #print(bootstrap_val)

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
            self.local_ac.advantages:advantages,
            self.local_ac.state_init:self.batch_rnn_state
        }

        self.batch_rnn_state, _ = sess.run([self.local_ac.state_out, self.local_ac.grad_apply], feed_dict=feed_dict)

    def work(self, gamma, sess, coord, max_ep_buffer_size=200, max_episode_count=300,skip_frame=3,max_episode_len=500):
        print("Starting worker {}".format(self.name))
        with sess.as_default():
            ep_count = 0
            while not coord.should_stop() and ep_count<max_episode_count:
                sess.run(self.update_local_ops)

                self.env.new_episode()
                s = self.env.get_state().screen_buffer
                s = process_frame(s)

                episode_buffer = []

                rnn_state = sess.run(self.local_ac.state_init)
                self.batch_rnn_state = rnn_state

                while not self.env.is_episode_finished():
                    a_dist,val, rnn_state = sess.run([self.local_ac.policy, self.local_ac.value, self.local_ac.state_out],
                                                feed_dict={
                                                    self.local_ac.inputs:[s],
                                                    self.local_ac.state_init:rnn_state
                                                })

                    #act_p = normalize_prob(a_dist[0])
                    act = np.random.choice(range(self.act_size),p=a_dist[0])
                    #act = np.argmax(a_dist==act)
                    r = self.env.make_action(self.actions[act], skip_frame)/max_episode_len # normalize to about 1.0
                    done = self.env.is_episode_finished()

                    if done:
                        s1 = s
                    else:
                        s1 = self.env.get_state().screen_buffer
                        s1 = process_frame(s1)

                    episode_buffer.append([s, act, r, s1, done, val[0,0]])

                    if len(episode_buffer) >= max_ep_buffer_size and not done:
                        v_pred = sess.run(self.local_ac.value,
                                          feed_dict={self.local_ac.inputs:[s], self.local_ac.state_init:rnn_state})[0,0]
                        self.train(episode_buffer, gamma,bootstrap_val=v_pred, sess=sess)
                        episode_buffer = []
                        sess.run(self.update_local_ops)

                    if done:
                        ep_count += 1
                        score = self.env.get_total_reward()
                        print("Episode {} finished with total reward: {}".format(ep_count, score))
                        sendStatElastic({"score": score,'game_name': 'VizDoom-Basic','episode':ep_count})
                        break

                if len(episode_buffer) != 0:
                    self.train(episode_buffer, gamma, 0.0, sess)


if __name__=="__main__":
    in_size = 84 * 84
    max_episode_len = 10000
    action_count = 3
    gamma = 0.99
    num_workers = multiprocessing.cpu_count() - 2
    #num_workers = 3
    print("Running with {} workers".format(num_workers))

    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    with tf.variable_scope("master"):
        master_network = ACNetwork(im_size=in_size, act_size=action_count, scope="master", trainer=None)

    workers = []
    for k in range(num_workers):
        w = Worker(in_size, action_count,k, trainer, get_doom_game())
        workers.append(w)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        worker_threads = []
        for wk in workers:
            work = lambda : wk.work(gamma, sess, coord, max_episode_count=max_episode_len)
            t = threading.Thread(target=(work))

            t.start()

            time.sleep(0.5)
            worker_threads.append(t)

        coord.join(worker_threads)
        print("Started all threads")
