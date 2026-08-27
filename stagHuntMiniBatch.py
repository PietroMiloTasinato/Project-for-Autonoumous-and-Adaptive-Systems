import gymnasium as gym
import gymnasium_stag_hunt
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from statistics import stdev, mean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

def distances(new_position: list):

    pos_1 = [new_position[0], new_position[1]]
    pos_2 = [new_position[2], new_position[3]]
    pos_s = [new_position[4], new_position[5]]

    
    dist_1s = np.sqrt( np.square(pos_1[0] - pos_s[0]) + np.square(pos_1[1] - pos_s[1]))
    dist_2s = np.sqrt( np.square(pos_2[0] - pos_s[0]) + np.square(pos_2[1] - pos_s[1]))
    dist_21 = np.sqrt( np.square(pos_2[0] - pos_1[0]) + np.square(pos_2[1] - pos_1[1]))
    
    return dist_1s, dist_2s, dist_21

def one_hot_encoder(input: list = [], entities: int = 0, dimX: int = 5, dimY: int = 5):
    ret = []

    for _ in range(entities):
        ret.append(np.zeros(dimX*dimY))
    
    # since the env is 2D, I will never have an odd number of coordinates
    for i in range((int)(len(input)*0.5)):
        ret[i][input[2*i]*dimX + input[2*i+1]] = 1
        #i+=1
    

    ret = tf.constant(ret, dtype=tf.float32)
    ret = tf.reshape(ret, [-1])
    ret = tf.keras.ops.concatenate([ret, tf.constant(distances(input))])

    #for shape consistency, we return a tensor
    return ret


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
    #Flatten(input_shape=(entities, 25)),
    Flatten(input_shape=(128,)),
    #Dense(128, activation = 'linear'),
    Dense(128, activation = 'relu'),
    Dense(64, activation = 'relu'),
    #Dense(64, activation = 'relu'),
    Dense(5, activation='softmax'),
    ])

# player 1 baseline model, smaller and simpler than policy model, we want a scalar output so output_dim = 1
pl1_bl_model = Sequential([
    #Flatten(input_shape=(entities, 25)),
    Flatten(input_shape=(128,)),
    Dense(128, activation = 'relu'),
    #Dense(64, activation = 'relu'),
    Dense(1, activation = 'linear'),
])

# player 2 policy model
pl2_pi_model = Sequential([
    #Flatten(input_shape=(entities, 25)),
    Flatten(input_shape=(128,)),
    #Dense(64, activation = 'linear'),
    Dense(128, activation = 'relu'),
    #Dense(64, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(5, activation='softmax'),
    ])

# player 2 baseline model, smaller and simpler than policy model, we want a scalar output so output_dim = 1
pl2_bl_model = Sequential([
    #Flatten(input_shape = (entities, 25)),
    Flatten(input_shape=(128,)),
    Dense(128, activation = 'relu'),
    #Dense(64, activation = 'relu'),
    Dense(1, activation = 'linear'),
])

# note: we use Adam because it implements
pl1_pi_opt = tf.keras.optimizers.Adam(1e-3)
pl1_bl_opt = tf.keras.optimizers.Adam(1e-2)
pl2_pi_opt = tf.keras.optimizers.Adam(1e-3)
pl2_bl_opt = tf.keras.optimizers.Adam(1e-2)

def train_baseline(bl_model, states, returns, bl_opt, epochs=8, batch_size=64):
    n = states.shape[0]
    bl_loss = 0
    for _ in range(epochs):

        idx = tf.random.shuffle(tf.range(n))
        for start in range(0, n, batch_size):

            b_idx = idx[start : (start + batch_size)]
            b_states = tf.gather(states, b_idx)
            b_returns = tf.gather(returns, b_idx)

            with tf.GradientTape() as bl_tape:
                bl_value = tf.squeeze(bl_model(b_states), axis=-1)
                bl_loss = tf.reduce_mean(tf.square(b_returns - bl_value))

            grads = bl_tape.gradient(bl_loss, bl_model.trainable_variables)
            bl_opt.apply_gradients(zip(grads, bl_model.trainable_variables))
    return bl_loss

def train_actor(pi_model, bl_model, states, actions, returns, pi_opt, entropy_coeff):
    
    with tf.GradientTape() as pi_tape:

        # forward pass for the two models
        bl_value = bl_model(states)  # approximation od the value function
        bl_value = tf.squeeze(bl_value, axis=-1)

        action_probs = pi_model(states)  # action probabilities for a given state and a fixed policy
                
        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs*tf.math.log(action_probs + 1e-8), axis = -1))

        # calculating advantage from the baseline model
        advantages = tf.stop_gradient(returns - bl_value)
        advantages = (advantages - mean(advantages.numpy())) / (stdev(advantages.numpy()) + 1e-8)
        #print("advantages for current epoch, mean: ", mean(advantages.numpy()), " and stddev: ", stdev(advantages.numpy()))

        # note: the masking is needed 
        action_masks = tf.one_hot(actions, action_probs.shape[-1])
        chosen_probs = tf.reduce_sum(action_probs * action_masks, axis=1)  
        log_probs = tf.math.log(chosen_probs + 1e-8)  # epsilon avoids errors due to ln(0)

        #compute loss
        pi_loss = -tf.reduce_mean(log_probs * advantages) # loss given by the REINFORCE with baseline formula
        tot_pi_loss = pi_loss - (entropy_coeff)*entropy
        #print(pi_loss)

    # Compute gradients
    pi_grads = pi_tape.gradient(tot_pi_loss, pi_model.trainable_variables)

    # Apply gradients 
    # note: zip is needed to "allign" the values from the gradients to the actual weights of the networks
    pi_opt.apply_gradients(zip(pi_grads, pi_model.trainable_variables))

    return tot_pi_loss

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


states, pl1_actions, pl2_actions, pl1_rewards, pl2_rewards, raw_pl1_rew, raw_pl2_rew = [], [], [], [], [], [], []
pl1_returns, pl2_returns, pl1_raw_ret, pl2_raw_ret = [], [], [], []

flip, active1, active2 = True, True, True

episode_len: int = 400
num_updates: int = 64
epoch_batch: int = 8

nrows = int(np.sqrt(num_updates))
ncols = nrows * 2
fig, ax = plt.subplots(nrows= nrows, ncols=ncols)

for epo in range(num_updates):

    sole_hunt_p1, sole_hunt_p2, plant_p1, plant_p2, hunt= 0, 0, 0, 0, 0

    #try scaling the mauling punishment to not discourage early exploration:
    progress = epo / num_updates
    mauling_punishment = -1.0 - (4.0 * progress)
    #mauling_punishment = -1
    stag_reward = 10.0 - (5.0 * progress)
    

    for i in range(epoch_batch):

        env = gym.make("StagHunt-Hunt-v0", obs_type='coords', stag_follows = False, stag_reward =stag_reward ,enable_multiagent = True, flip_obs = False, max_timesteps=10000, run_away_after_maul=True, mauling_punishment=mauling_punishment)  # you can pass config parameters here
        #initialization step
        obs, info = env.reset()

        old_rew = [0]*3
        old_rew[0], old_rew[1], old_rew[2] = (distances(obs[0]))

        state = one_hot_encoder(obs[0], entities, 5, 5)

        for epi in range(episode_len):    

            state = one_hot_encoder(obs[0], entities, 5, 5)
            states.append(state)

            pl1_action = select_action(pl1_pi_model, state)
            pl2_action = select_action(pl2_pi_model, state)

            obs, reward, done, trunc, info = env.step([pl1_action, pl2_action])
            #obs is a tensor, rewards is a tuple info is a dict, done and trunc are bool

            if(done or trunc):
                print("max number of moves reached")
                exit(1)


            pl1_actions.append(pl1_action)
            pl2_actions.append(pl2_action)

            current_rew = [0]*3
            current_rew[0], current_rew[1], current_rew[2] = distances(obs[0])
            shaping_factor = -(0.5*((0.99*current_rew[0] - old_rew[0])+(0.99*current_rew[1] - old_rew[1])+(0.99*current_rew[2] - old_rew[2])))
            old_rew = current_rew

            if(isinstance(reward, tuple)):

                #print("distance of 1: ", dist_1, " distance of 2: ", dist_2, " max dist: ", shaping_factor)

                pl1_rewards.append(reward[0] + shaping_factor)
                pl2_rewards.append(reward[1] + shaping_factor)
                raw_pl1_rew.append(reward[0])
                raw_pl2_rew.append(reward[1])

                if(reward[0]== 1): plant_p1 += 1
                if(reward[0]== mauling_punishment): sole_hunt_p1 += 1
                if(reward[0]== stag_reward): hunt += 1
                if(reward[1]== 1): plant_p2 += 1
                if(reward[1]== mauling_punishment): sole_hunt_p2 += 1
        
            else: 
                print("rewards error")
                exit(1)

            #obs, rewards, done, trunc, info = env.step([env.action_space.sample(), env.action_space.sample()])
            #env.render()
        
        #if(i == 1): sanity_check = compute_returns(pl1_rewards, 0.99)
        pl1_returns.append(compute_returns(pl1_rewards, 0.99))
        pl2_returns.append(compute_returns(pl2_rewards, 0.99))

        pl1_raw_ret.append(compute_returns(raw_pl1_rew, 0.99))
        pl2_raw_ret.append(compute_returns(raw_pl2_rew, 0.99))

        pl1_rewards.clear()
        pl2_rewards.clear()
        raw_pl1_rew.clear()
        raw_pl2_rew.clear()
                    
        env.close()

    stacked_states= tf.stack(states)
    #print("stacked states shape:", stacked_states.shape)
    #for j in range(len(sanity_check)):
    #    if(not(sanity_check[j] == pl1_returns[-1][j])): print("error: ", sanity_check[j], " vs ", pl1_returns[-1][j])
    
    pl_returns_ten = tf.keras.ops.concatenate(pl1_returns, axis =-1)
    pl_raw_ret_ten = tf.keras.ops.concatenate(pl1_raw_ret, axis =-1)

    entropy_coeff = (1e-3)/(1+hunt)

    pi_loss = train_actor(pl1_pi_model, pl1_bl_model, states=stacked_states, actions=pl1_actions, returns=pl_returns_ten, pi_opt=pl1_pi_opt, entropy_coeff=entropy_coeff)
    bl_loss = train_baseline(pl1_bl_model, states=stacked_states, returns = pl_returns_ten, bl_opt = pl1_bl_opt)
    print("for epoch: ", epo, " for player 1 we have a policy loss: ", pi_loss, " and a baseline loss: ", bl_loss)

    bl_value = tf.squeeze(pl1_bl_model(stacked_states), axis =-1)
    
    print("explained variance for current epoch and player 1:" ,1 - tf.math.reduce_variance(pl_returns_ten - bl_value) / tf.math.reduce_variance(pl_returns_ten))
    #print("explained variance for current epoch and player 1, raw returns:" ,1 - tf.math.reduce_variance(pl_raw_ret_ten - bl_value) / tf.math.reduce_variance(pl_raw_ret_ten))
    ax[(epo%nrows), 2*int(epo/nrows)].scatter(pl_returns_ten, bl_value)


    pl_returns_ten = tf.keras.ops.concatenate(pl2_returns, axis =-1)
    pl_raw_ret_ten = tf.keras.ops.concatenate(pl2_raw_ret, axis =-1)

    pi_loss = train_actor(pl2_pi_model, pl2_bl_model, states=stacked_states, actions=pl2_actions, returns=pl_returns_ten, pi_opt=pl2_pi_opt, entropy_coeff=entropy_coeff)
    bl_loss = train_baseline(pl2_bl_model, states=stacked_states, returns = pl_returns_ten, bl_opt = pl2_bl_opt)
    print("for epoch: ", epo, " for player 2 we have a policy loss: ", pi_loss, " and a baseline loss: ", bl_loss)

    bl_value = tf.squeeze(pl2_bl_model(stacked_states), axis =-1)
    print("explained variance for current epoch and player 2:" ,1 - tf.math.reduce_variance(pl_returns_ten - bl_value) / tf.math.reduce_variance(pl_returns_ten))
    #print("explained variance for current epoch and player 2, raw returns:" ,1 - tf.math.reduce_variance(pl_raw_ret_ten - bl_value) / tf.math.reduce_variance(pl_raw_ret_ten))
    ax[(epo%nrows), 2*int(epo/(nrows))+1].scatter(pl_returns_ten, bl_value)


    print("nuber of successful hunts: ", hunt, " number of unsuccessful hunts for player 1: ", sole_hunt_p1, " and for player 2: ", sole_hunt_p2)
    print("number of times player 1 harvested plants: ", plant_p1, " number of times player 2 harvested plants:", plant_p2)
    print("reward for pl1:", (hunt*5 +plant_p1*1 - sole_hunt_p1*5)/epoch_batch)
    print("reward for pl2:", (hunt*5 +plant_p2*1 - sole_hunt_p2*5)/epoch_batch)


    """
    tot1, tot2 = 0, 0
    for t in reversed(pl1_rewards):
        #print("reward for player 1: ", t)
        tot1 += t
    for t in reversed(pl2_rewards):
        #print("reward for player 2: ", t)
        tot2 += t

    print("epoch: ", epo, " player 1 total reward: ", tot1)
    print("epoch: ", epo, " player 2 total reward: ", tot2)
    """
    pl_returns_ten = 0
    pl_raw_ret_ten = 0

    states.clear()
    stacked_states = tf.keras.ops.empty((2, 2))
    pl1_actions.clear()
    pl2_actions.clear()
    pl1_returns.clear()
    pl2_returns.clear()
    pl1_raw_ret.clear()
    pl2_raw_ret.clear()

    print("\n")

plt.show()