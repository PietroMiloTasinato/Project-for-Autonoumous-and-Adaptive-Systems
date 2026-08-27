import gymnasium as gym
import gymnasium_stag_hunt
import time
import tensorflow as tf
import numpy as np
from statistics import stdev, mean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

def one_hot_encoder(input: list = [], entities: int = 0, dimX: int = 5, dimY: int = 5):
    ret = []

    for _ in range(entities):
        ret.append(np.zeros(dimX*dimY))
    
    # since the env is 2D, I will never have an odd number of coordinates
    for i in range((int)(len(input)*0.5)):
        ret[i][input[2*i]*dimX + input[2*i+1]] = 1
        #i+=1
    
    #for shape consistency, we return a tensor
    return tf.constant(ret)

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
    #Dense(128, activation = 'linear'),
    Dense(128, activation = 'relu'),
    #Dense(64, activation = 'linear'),
    Dense(64, activation = 'relu'),
    Dense(5, activation='softmax'),
    ])

# player 1 baseline model, smaller and simpler than policy model, we want a scalar output so output_dim = 1
pl1_bl_model = Sequential([
    Flatten(input_shape=(entities, 25)),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(1, activation = 'linear'),
])

# player 2 policy model
pl2_pi_model = Sequential([
    Flatten(input_shape=(entities, 25)),
    #Dense(64, activation = 'linear'),
    Dense(128, activation = 'relu'),
    #Dense(64, activation = 'linear'),
    Dense(64, activation = 'relu'),
    Dense(5, activation='softmax'),
    ])

# player 2 baseline model, smaller and simpler than policy model, we want a scalar output so output_dim = 1
pl2_bl_model = Sequential([
    Flatten(input_shape = (entities, 25)),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(1, activation = 'linear'),
])

# note: we use Adam because it implements
pl1_pi_opt = tf.keras.optimizers.Adam(1e-3)
pl1_bl_opt = tf.keras.optimizers.Adam(1e-2)
pl2_pi_opt = tf.keras.optimizers.Adam(1e-3)
pl2_bl_opt = tf.keras.optimizers.Adam(1e-2)


def training_step(pi_model, bl_model, states, actions, returns, pi_opt, bl_opt):
    
    with tf.GradientTape() as pi_tape, tf.GradientTape() as bl_tape:

        # forward pass for the two models
        bl_value = bl_model(states)  # approximation od the value function
        bl_value = tf.squeeze(bl_value, axis=-1)

        action_probs = pi_model(states)  # action probabilities for a given state and a fixed policy

        print("explained variance for current epoch and current player:" ,1 - tf.math.reduce_variance(returns - bl_value) / tf.math.reduce_variance(returns))
        
        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs*tf.math.log(action_probs + 1e-8), axis = -1))

        print("shapes: ", bl_value.shape, returns.shape)

        # calculating advantage from the baseline model
        advantages = tf.stop_gradient(returns - bl_value)
        print("advantages for current epoch, mean: ", mean(advantages.numpy()), " and stddev: ", stdev(advantages.numpy()))

        # note: the masking is needed 
        action_masks = tf.one_hot(actions, action_probs.shape[-1])
        chosen_probs = tf.reduce_sum(action_probs * action_masks, axis=1)  
        log_probs = tf.math.log(chosen_probs + 1e-8)  # epsilon avoids errors due to ln(0)

        #compute loss
        pi_loss = -tf.reduce_mean(log_probs * advantages) # loss given by the REINFORCE with baseline formula
        tot_pi_loss = pi_loss - (1e-3)*entropy
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
    
    return tf.cast(tf.constant(returns), dtype=tf.float32)
 
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

states, pl1_actions, pl2_actions, pl1_rewards, pl2_rewards = [], [], [], [], []


flip, active1, active2 = True, True, True

episode_len: int = 400
num_updates: int = 40

for epo in range(num_updates):

    sole_hunt_p1, sole_hunt_p2, plant_p1, plant_p2, hunt= 0, 0, 0, 0, 0

    env = gym.make("StagHunt-Hunt-v0", obs_type='coords', stag_follows = False, stag_reward =5.0 ,enable_multiagent = True, flip_obs = False, max_timesteps=10000)  # you can pass config parameters here
    #initialization step
    obs, info = env.reset()

    state = one_hot_encoder(obs[0], entities, 5, 5)

    for epi in range(episode_len):    

        state = one_hot_encoder(obs[0], entities, 5, 5)
        states.append(state)

        pl1_action = select_action(pl1_pi_model, state)
        pl2_action = select_action(pl2_pi_model, state)

        obs, reward, done, trunc, info = env.step([pl1_action, pl2_action])
        #obs is a shape, rewards is a tuple info is a dict, done and trunc are bool

        if(done or trunc):
            print("max number of moves reached")
            exit(1)


        pl1_actions.append(pl1_action)
        pl2_actions.append(pl2_action)
        

        if(isinstance(reward, tuple)):

            
            pl1_rewards.append(reward[0])
            pl2_rewards.append(reward[1])

            if(reward[0]== 1): plant_p1 += 1
            if(reward[0]== -5): sole_hunt_p1 += 1
            if(reward[0]== 5): hunt += 1
            if(reward[1]== 1): plant_p2 += 1
            if(reward[1]== -5): sole_hunt_p2 += 1
    
        else: 
            print("rewards error")
            exit(1)

        #obs, rewards, done, trunc, info = env.step([env.action_space.sample(), env.action_space.sample()])
        #env.render()
    
    stacked_states= tf.stack(states)
    #print("stacked states shape:", stacked_states.shape)
    

    if (active1):
        pi_loss, bl_loss = training_step(pl1_pi_model, pl1_bl_model, states=stacked_states, actions=pl1_actions, returns=compute_returns(pl1_rewards, 0.99), pi_opt=pl1_pi_opt, bl_opt=pl1_bl_opt)
        print("for epoch: ", epo, " for player 1 we have a policy loss: ", pi_loss, " and a baseline loss: ", bl_loss)
    if(active2): 
        pi_loss, bl_loss = training_step(pl2_pi_model, pl2_bl_model, states=stacked_states, actions=pl2_actions, returns=compute_returns(pl2_rewards, 0.99), pi_opt=pl2_pi_opt, bl_opt=pl2_bl_opt)
        print("for epoch: ", epo, " for player 2 we have a policy loss: ", pi_loss, " and a baseline loss: ", bl_loss)

    """
    #freezing window implementation
    window_len: int = 8
    if(epo % window_len == 0):
        flip = not(flip)
    else:
        if(flip):
            active1 = True
            active2 = False
        else:
            active1 = False
            active2 = True
    """

    print("nuber of successful hunts: ", hunt, " number of unsuccessful hunts for player 1: ", sole_hunt_p1, " and for player 2: ", sole_hunt_p2)
    print("number of times player 1 harvested plants: ", plant_p1, " number of times player 2 harvested plants:", plant_p2)
    

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

