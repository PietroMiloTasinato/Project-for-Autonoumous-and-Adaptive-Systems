# Common Critic idea inspired by: https://arxiv.org/pdf/1705.08926
# note, the implementation for multi input and multi output follows: https://stackoverflow.com/questions/66845924/multi-input-multi-output-model-with-keras-functional-api

import gymnasium as gym
import gymnasium_stag_hunt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, concatenate

def one_hot_encoder(input: list = [], entities: int = 0, dimX: int = 5, dimY: int = 5):
    ret = []

    for e in range(entities):
        ret.append(np.zeros(dimX*dimY))
    
    # since the env is 2D, I will never have an odd number of coordinates
    for i in range((int)(entities)):
        ret[i][input[2*i]*dimX + input[2*i+1]] = 1
    
    #for shape consistency, we return a tensor
    return tf.constant(ret)


def distance_reward(old_position: list, new_position: list):
    old_pos_1 = [old_position[0], old_position[1]]
    old_pos_2 = [old_position[2], old_position[3]]
    
    new_pos_1 = [new_position[0], old_position[1]]
    new_pos_2 = [new_position[2], new_position[3]]
    
    old_pos_s = [old_position[4], old_position[5]]
    new_pos_s = [new_position[4], new_position[5]]
    
    rew_1, rew_2 = False, False

    if(np.sqrt( np.square(old_pos_1[0] - old_pos_s[0]) + np.square(old_pos_1[1] - old_pos_s[1])) > np.sqrt( np.square(new_pos_1[0] - new_pos_s[0]) + np.square(new_pos_1[1] - new_pos_s[1]))):
        rew_1 = True

    if(np.sqrt( np.square(old_pos_2[0] - old_pos_s[0]) + np.square(old_pos_2[1] - old_pos_s[1])) > np.sqrt( np.square(new_pos_2[0] - new_pos_s[0]) + np.square(new_pos_2[1] - new_pos_s[1]))):
        rew_2 = True

    return rew_1, rew_2


def one_hot_action(input: int = 0, n_actions: int = 0):

    ret = np.zeros(n_actions)
    ret[input] = 1

    return tf.constant(ret)

#Move encoding for stag hunt:
n_moves = 5
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAND = 4

#one hot encoding: 5 entities: player1, player2, stag, 2 plants
entities = 5

# player 1 policy model
pl1_pi_model = Sequential([
    Flatten(input_shape=(entities, 25)),
    Dense(64, activation = 'linear'),
    Dense(64, activation = 'tanh'),
    Dense(64, activation = 'linear'),
    Dense(64, activation = 'tanh'),
    Dense(5, activation='softmax'),
    ])

# player 2 policy model
pl2_pi_model = Sequential([
    Flatten(input_shape=(entities, 25)),
    Dense(64, activation = 'linear'),
    Dense(64, activation = 'tanh'),
    Dense(64, activation = 'linear'),
    Dense(64, activation = 'tanh'),
    Dense(5, activation='softmax'),
    ])

"""
common_critic = Sequential([
    Flatten(input_shape = (entities, 25)),
    Dense(64, activation = 'relu'),
    Dense(64, activation = 'relu'),
    ], 
    [Dense(1, activation = 'linear', name = 'critic_1'), Dense(1, activation = 'linear', name = 'critic_2')]
    )
"""

#common critic model
input = Input(shape=(entities , 25))
#input = Flatten()(input)
#action_1 = Input(shape = (n_moves,))
#action_2 = Input(shape = (n_moves,))

#x = concatenate([input, action_1, action_2])
#x = Dense(128, activation = 'tanh')(x)
x = Dense(128, activation = 'tanh')(input)
x = Dense(128, activation = 'tanh')(x)
x = Dense(64, activation = 'tanh')(x)
x = Dense(64, activation = 'tanh')(x)

critic_1 = Dense(1, activation = 'linear')(x)
critic_2 = Dense(1, activation = 'linear')(x)

common_critic = Model( inputs = [input], outputs = [critic_1, critic_2], name="common_critic")
#common_critic = Model( inputs = [input, action_1, action_2], outputs = [critic_1, critic_2], name="common_critic")

# note: we use Adam because it implements
p1_opt = tf.keras.optimizers.Adam(0.01)
p2_opt = tf.keras.optimizers.Adam(0.01)
cc_opt = tf.keras.optimizers.Adam(0.01)


def training_step(p1_model, p2_model, cc_model, states, actions_p1, actions_p2, returns, p1_opt, p2_opt, cc_opt):

    with tf.GradientTape() as p1_tape, tf.GradientTape() as cc_tape, tf.GradientTape() as p2_tape:

        # forward pass for the two models
        #states = tf.reshape(states, [tf.shape(states)[0], -1])
        #cc_v1, cc_v2 = cc_model([states, actions_p1, actions_p2])  # approximation od the value function
        cc_v1, cc_v2 = cc_model(states)
        
        action_probs_p1 = p1_model(states)  # action probabilities for a given state and a fixed policy
        action_probs_p2 = p2_model(states)

        entropy_p1 = -tf.reduce_sum(action_probs_p1*tf.math.log(action_probs_p1))
        entropy_p2 = -tf.reduce_sum(action_probs_p2*tf.math.log(action_probs_p2))

        # calculating advantage from the baseline model
        #bl_v = tf.squeeze(bl_value, axis=-1)
        advantages_p1 = returns[0] - cc_v1
        advantages_p1 = tf.stop_gradient(advantages_p1)
        advantages_p2 = returns[1] - cc_v2
        advantages_p2 = tf.stop_gradient(advantages_p2)
        #print(advantages)

        action_masks = tf.one_hot(actions_p1, action_probs_p1.shape[-1])
        chosen_probs_p1 = tf.reduce_sum(action_probs_p1 * action_masks, axis=1)  
        log_probs_p1 = tf.math.log(chosen_probs_p1 + 1e-8)  # epsilon avoids errors due to ln(0)

        action_masks = tf.one_hot(actions_p2, action_probs_p2.shape[-1])
        chosen_probs_p2 = tf.reduce_sum(action_probs_p2 * action_masks, axis=1)
        log_probs_p2 = tf.math.log(chosen_probs_p2 + 1e-8) 


        #compute loss
        p1_loss = -tf.reduce_mean(log_probs_p1 * advantages_p1) 
        tot_p1_loss = p1_loss - 0.05*entropy_p1
        p2_loss = -tf.reduce_mean(log_probs_p2 * advantages_p2) 
        tot_p2_loss = p2_loss - 0.05*entropy_p2

        cc_loss = tf.reduce_mean(tf.square(cc_v1 - returns[0])) + tf.reduce_mean(tf.square(cc_v2 - returns[1])) # MSE
        
    # Compute gradients
    p1_grads = p1_tape.gradient(tot_p1_loss, p1_model.trainable_variables)
    p2_grads = p2_tape.gradient(tot_p2_loss, p2_model.trainable_variables)
    cc_grads = cc_tape.gradient(cc_loss, cc_model.trainable_variables)


    # Apply gradients 
    # note: zip is needed to "allign" the values from the gradients to the actual weights of the networks
    p1_opt.apply_gradients(zip(p1_grads, p1_model.trainable_variables))
    p2_opt.apply_gradients(zip(p2_grads, p2_model.trainable_variables))
    cc_opt.apply_gradients(zip(cc_grads, cc_model.trainable_variables))

    return tot_p1_loss, tot_p2_loss, cc_loss

def select_action(pi_model, state):
    
    state = tf.expand_dims(state, axis=0)
    action_probs = pi_model(state)
    action_probs = tf.squeeze(action_probs, axis=0) 
    
    action = tf.random.categorical(tf.math.log(action_probs[tf.newaxis, :]), num_samples=1)
    action = int(action[0, 0])

    #print("action taken: ", action)
    
    return action

def compute_returns(rewards, discount_rate):
    
    returns = []
    cm_rew = 0   # discounted cumulative return

    for r in reversed(rewards):
        cm_rew = r + discount_rate * cm_rew
        returns.insert(0, cm_rew)
    
    return returns
 
"""
        # Env Specific
        "env_config": {
            "obs_type": "image",
            "forage_reward": 1.0,
            "stag_reward": 5.0,
            "stag_follows": False,
            "mauling_punishment": -.5,
            "enable_multiagent": True,
        }
"""

env = gym.make("StagHunt-Hunt-v0", obs_type='coords', stag_follows = False, stag_reward = 5.0, enable_multiagent = True, flip_obs = False, max_timesteps=10000)  # you can pass config parameters here
#initialization step
obs, info = env.reset()

states, pl1_rewards, pl2_rewards, pl1_actions, pl2_actions = [], [], [], [], []

counter: int = 0
s1, s2 = 0,0
active1, active2 = True, True

episode: int = 10
epoch: int = 20

state = one_hot_encoder(obs[0], entities, 5, 5)
old_pos = obs[0]

for epo in range(epoch):
    for epi in range(episode):    
        
        pl1_action = select_action(pl1_pi_model, state)
        pl2_action = select_action(pl2_pi_model, state)

        obs, reward, done, trunc, info = env.step([pl1_action, pl2_action])
        #obs is a tensor, rewards is a tuple info is a dict, done and trunc are bool

        if(done or trunc): break

        state = one_hot_encoder(obs[0], entities, 5, 5)
        
        states.append(state)
        pl1_actions.append(pl1_action)
        pl2_actions.append(pl2_action)
        
        rew_1, rew_2 = distance_reward(old_pos, obs[0])
        

        if(isinstance(reward, tuple)):

            pl1_rewards.append(reward[0])            
            pl2_rewards.append(reward[1])
    
        else: 
            print("rewards error")
            exit(1)
        
        #env.render()
    
    stacked_states= tf.stack(states)
    #print("stacked states shape:", stacked_states.shape)
    


    p1_loss, p2_loss, cc_loss = training_step(pl1_pi_model, pl2_pi_model, common_critic, states=stacked_states, actions_p1=pl1_actions, actions_p2=pl2_actions, returns=[compute_returns(pl1_rewards, 0.99), compute_returns(pl2_rewards, 0.99)], p1_opt=p1_opt, p2_opt= p2_opt, cc_opt=cc_opt)
    print("for epoch: ", epo, " for player 1 we have a policy loss: ", p1_loss, " for player 2 we have a policy loss: ", p2_loss, " and a baseline loss: ", cc_loss)

    tot1, tot2 = 0, 0
    for t in reversed(pl1_rewards):
        #print("reward for player 1: ", t)
        tot1 += t
    for t in reversed(pl2_rewards):
        #print("reward for player 2: ", t)
        tot2 += t

    print("epoch: ", epo, " player 1 total reward: ", tot1)
    print("epoch: ", epo, " player 2 total reward: ", tot2)
    
    states.clear()
    stacked_states = tf.keras.ops.empty((2, 2))
    pl1_actions.clear()
    pl2_actions.clear()
    pl1_rewards.clear()
    pl2_rewards.clear()
    
    episode += 10

env.close()

