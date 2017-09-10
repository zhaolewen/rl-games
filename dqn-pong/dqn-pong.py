import numpy as np
import gym


in_dim = 80 * 80 # input dimension
hidden_dim = 200 # hidden layer dimension
gamma = 0.99 # reward discount factor

# model
model = {}
model['W1'] = np.random.randn(hidden_dim, in_dim)/ np.sqrt(in_dim) # "Xavier" initialization
model["W2"] = np.random.randn(hidden_dim) / np.sqrt(hidden_dim)

def discounted_reward(r):
    # calculate discounted reward
    disc_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(len(r))):
        if r[t] != 0: # end of game
            running_add = 0
        running_add = running_add * gamma + r[t]
        disc_r[t] = running_add

    return disc_r


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def policy_forward(x):
    h = np.dot(model["W1"], x)
    h[h<0] = 0 # ReLU
    logp = np.dot(model["W2"], h)
    p = sigmoid(logp)

    return p, h # prob of taking action 2: go up, and hidden state

def pre_process_image(img):
    # from 210*160*3 uint8 to 6400 (80*80) 1D float vector
    img = img[35:195]
    img = img[::2,::2,0] # downsample by 2
    img[img == 144] = 0
    img[img == 109] = 0 # erase 2 types of background
    img[img != 0] = 1

    return img.astype(np.float).ravel() # to float 1D

env = gym.make("Pong-v0")
observation = env.reset()
img_prev = None
reward_sum = 0
episode_count = 0
rew_list = []
dlogp_list = []

while True:
    env.render()

    # calculate difference
    img_curr = pre_process_image(observation)
    if img_prev is None:
        img_diff = np.zeros(in_dim)
    else:
        img_diff = img_curr - img_prev
    img_prev = img_curr

    act_prob, hid = policy_forward(img_diff)
    action = 2 if np.random.uniform() < act_prob else  3
    # 2 up, 3 down

    y = 1 if action==2 else 0
    dlogp_list.append(y - act_prob) # grad that encourage taking the action that is taken

    observation,reward, done, info = env.step(action)
    reward_sum += reward
    rew_list.append(reward)

    if done:
        episode_count += 1

        stack_rew = np.vstack(rew_list)
        disc_rew = discounted_reward(stack_rew)
        disc_rew -= np.mean(disc_rew)
        disc_rew /= np.std(disc_rew)
        print(disc_rew)

        stack_dlogp = np.vstack(dlogp_list)
        stack_dlogp *= disc_rew


        print("Episode {} done".format(episode_count))
        observation = env.reset()
        img_prev = None