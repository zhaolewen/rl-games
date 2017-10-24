import gym
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np
import scipy.misc
import time, requests
import PIL
from model import QNetwork,ExperienceBuffer,FrameBuffer

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
        op_holder.append(tf.assign(to_v, from_v.value() * tau + to_v.value() * (1-tau)))

    return op_holder

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

def discounted_reward(rs, gamma):
    total = 0
    for k in reversed(range(len(rs))):
        total = total * gamma + rs[k]

    return total

if __name__=="__main__":
    game_name = 'Breakout-v0'
    env = gym.make(game_name)
    game_name += '-ddqn-cnn'

    batch_size = 32 # num of experience traces
    update_target_step = 10000

    gamma = 0.99 # discount factor for reward
    e_start = 1.0 # prob of random action
    e_end = 0.1
    annel_steps  = 100000 # steps from e_start to e_end
    total_episodes = 10000
    update_step = 4

    pre_train_steps = 5000 # steps of random action before training begins
    logdir = "./checkpoints/ddqn-cnn"

    h_size = 512
    action_size = env.action_space.n
    skip_frame = 4
    frame_count = 4
    img_size = 84

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

    with sv.managed_session() as sess:
        update_qn_op = update_target_graph(scope_main, scope_target)
        step_value = sess.run(global_step)

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
                env.render()
                if total_step%skip_frame !=0:
                    s1, reward, done, obs = env.step(last_act)
                    last_frame = s1
                else:
                    # normal process
                    begin_frames = frame_buffer.frames()
                    act,_ = main_qn.predict_act(begin_frames, session=sess)
                    act = act[0]
                    if np.random.rand() < e or total_step<pre_train_steps:
                        act = np.random.randint(0, action_size)

                    last_act = act

                    s1, reward, done, _ = env.step(act)

                    r2 = clip_reward(reward)
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

                    s = s1
                    s_frame = s1_frame

                ep_rewards.append(reward)
                total_step += 1

                if total_step % update_target_step == 0:
                    sess.run(update_qn_op)

                if done:
                    disc_r = discounted_reward(ep_rewards, gamma)
                    score = discounted_reward(ep_rewards, 1)

                    print("Episode {} finished in {} seconds with discounted reward {}, score {}, e {}, global step {}".format(ep, time.time()-t_ep_start, disc_r, score,e, step_value))
                    sendStatElastic({"discount_reward":disc_r, "score":score,"episode":ep,"rand_e_prob":e,'game_name':game_name})
                    break