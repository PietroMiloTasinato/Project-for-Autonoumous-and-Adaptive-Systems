import gymnasium as gym
import gymnasium_stag_hunt
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

def one_hot_encoder(input: list = [], entities: int = 0, dimX: int = 5, dimY: int = 5):
    ret = []

    for e in range(entities):
        ret.append(np.zeros(dimX*dimY))
    
    # since the env is 2D, I will never have an odd number of coordinates
    for i in range((int)(len(input)*0.5)):
        ret[i][input[2*i]*dimX + input[2*i+1]] = 1
        #i+=1
    
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



#Move encoding for stag hunt:

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

# player 1 baseline model, smaller and simpler than policy model, we want a scalar output so output_dim = 1
pl1_bl_model = Sequential([
    Flatten(input_shape=(entities, 25)),
    Dense(64, activation = 'tanh'),
    Dense(1, activation = 'linear'),
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

# player 2 baseline model, smaller and simpler than policy model, we want a scalar output so output_dim = 1
pl2_bl_model = Sequential([
    Flatten(input_shape = (entities, 25)),
    Dense(64, activation = 'tanh'),
    Dense(1, activation = 'linear'),
])

# note: we use Adam because it implements 
pl1_pi_opt = tf.keras.optimizers.Adam(0.01)
pl1_bl_opt = tf.keras.optimizers.Adam(0.01)
pl2_pi_opt = tf.keras.optimizers.Adam(0.01)
pl2_bl_opt = tf.keras.optimizers.Adam(0.01)


def training_step(pi_model, bl_model, states, actions, returns, pi_opt, bl_opt):

    with tf.GradientTape() as pi_tape, tf.GradientTape() as bl_tape:

        # forward pass for the two models
        bl_value = bl_model(states)  # approximation od the value function
        
        action_probs = pi_model(states)  # action probabilities for a given state and a fixed policy
        
        entropy = -tf.reduce_sum(action_probs*tf.math.log(action_probs))

        # calculating advantage from the baseline model
        bl_value = tf.squeeze(bl_value, axis=-1)
        advantages = returns - bl_value
        advantages = tf.stop_gradient(advantages)
        #print(advantages)

        # note: the masking is needed 
        action_masks = tf.one_hot(actions, action_probs.shape[-1])
        chosen_probs = tf.reduce_sum(action_probs * action_masks, axis=1)  
        log_probs = tf.math.log(chosen_probs + 1e-8)  # epsilon avoids errors due to ln(0)

        #compute loss
        pi_loss = -tf.reduce_mean(log_probs * advantages) # loss given by the REINFORCE with baseline formula
        tot_pi_loss = pi_loss - 0.05*entropy
        bl_loss = tf.reduce_mean(tf.square(returns - bl_value)) # MSE
        #print(pi_loss, bl_loss)

    # Compute gradients
    pi_grads = pi_tape.gradient(tot_pi_loss, pi_model.trainable_variables)
    bl_grads = bl_tape.gradient(bl_loss, bl_model.trainable_variables)

    # Apply gradients 
    # note: zip is needed to "allign" the values from the gradients to the actual weights of the networks
    pi_opt.apply_gradients(zip(pi_grads, pi_model.trainable_variables))
    bl_opt.apply_gradients(zip(bl_grads, bl_model.trainable_variables))

    return tot_pi_loss, bl_loss

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

env = gym.make("StagHunt-Hunt-v0", obs_type='coords', stag_follows = False, stag_reward = 10.0 ,enable_multiagent = True, flip_obs = False, max_timesteps=10000)  # you can pass config parameters here
#initialization step
obs, info = env.reset()

states, pl1_actions, pl2_actions, pl1_rewards, pl2_rewards = [], [], [], [], []

counter: int = 0
s1, s2 = 0,0
active1, active2 = True, True

episode: int = 70
epoch: int = 15

state = one_hot_encoder(obs[0], entities, 5, 5)

old_pos = obs[0]

for epo in range(epoch):
    for epi in range(episode):    
        
        pl1_action = select_action(pl1_pi_model, state)
        pl2_action = select_action(pl2_pi_model, state)

        obs, reward, done, trunc, info = env.step([pl1_action, pl2_action])
        #obs is a tensor, rewards is a tuple info is a dict, done and trunc are bool

        if(done or trunc):
            print("max number of moves reached")
            exit(1)

        state = one_hot_encoder(obs[0], entities, 5, 5)

        states.append(state)
        pl1_actions.append(pl1_action)
        pl2_actions.append(pl2_action)
        
        rew_1, rew_2 = distance_reward(old_pos, obs[0])
        old_pos = obs[0]

        if(isinstance(reward, tuple)):

            if(rew_1):
                pl1_rewards.append(reward[0])
            else:
                pl1_rewards.append(reward[0])
            
            if(rew_2):
                pl2_rewards.append(reward[1])
            else:
                pl2_rewards.append(reward[1])
    
        else: 
            print("rewards error")
            exit(1)

        #obs, rewards, done, trunc, info = env.step([env.action_space.sample(), env.action_space.sample()])
        #env.render()
    
    stacked_states= tf.stack(states)
    #print("stacked states shape:", stacked_states.shape)
    
    """
    if(epo % 3 == 0):
        active1 = True
        active2 = True
    else:
        if(s1 < 2 and s2 > 0):
            active1 = True
            active2 = False
            s1 += 1
            s2 -= 1
        else:
            active1 = False
            active2 = True
            s1 -= 1
            s2 += 1
"""

    if (active1):
        pi_loss, bl_loss = training_step(pl1_pi_model, pl1_bl_model, states=stacked_states, actions=pl1_actions, returns=compute_returns(pl1_rewards, 0.99), pi_opt=pl1_pi_opt, bl_opt=pl1_bl_opt)
        print("for epoch: ", epo, " for player 1 we have a policy loss: ", pi_loss, " and a baseline loss: ", bl_loss)
    if(active2): 
        pi_loss, bl_loss = training_step(pl2_pi_model, pl2_bl_model, states=stacked_states, actions=pl2_actions, returns=compute_returns(pl2_rewards, 0.99), pi_opt=pl2_pi_opt, bl_opt=pl2_bl_opt)
        print("for epoch: ", epo, " for player 2 we have a policy loss: ", pi_loss, " and a baseline loss: ", bl_loss)
    
    

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

env.close()

