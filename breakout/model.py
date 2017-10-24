import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import random

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
            loss = tf.losses.huber_loss(self.target_q, Q)

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
        self.frames = []
        self.buffer_size = buffer_size

    def add(self, last_frame, new_frame, action, reward, done):
        self.buffer.append((action, reward,done))
        self.frames.append((last_frame, new_frame))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            self.frames.pop(0)

    def sample(self, batch_size, trace_len=4):
        idx = random.randint(0, len(self.buffer)-trace_len)
        dt_frames = [[self.frames[i-trace_len+k] for k in range(trace_len)] for i in idx]
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