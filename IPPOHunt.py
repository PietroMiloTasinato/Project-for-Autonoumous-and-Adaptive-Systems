import gymnasium as gym
import gymnasium_stag_hunt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

def distances(new_position: list):

    pos_1 = [new_position[0], new_position[1]]
    pos_2 = [new_position[2], new_position[3]]
    pos_s = [new_position[4], new_position[5]]

    
    dist_1s = np.sqrt( np.square(int(pos_1[0]) - int(pos_s[0])) + np.square(int(pos_1[1]) - int(pos_s[1])))
    dist_2s = np.sqrt( np.square(int(pos_2[0]) - int(pos_s[0])) + np.square(int(pos_2[1]) - int(pos_s[1])))
    dist_21 = np.sqrt( np.square(int(pos_2[0]) - int(pos_1[0])) + np.square(int(pos_2[1]) - int(pos_1[1])))
    
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

#one hot encoding: 5 entities: player1, player2, stag
entities = 5
gamma = 0.99

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
    Flatten(input_shape=(129,)),
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
    Flatten(input_shape=(129,)),
    Dense(128, activation = 'relu'),
    #Dense(64, activation = 'relu'),
    Dense(1, activation = 'linear'),
])

# note: we use Adam because it implements 
pl1_pi_opt = tf.keras.optimizers.Adam(3e-4)
pl1_bl_opt = tf.keras.optimizers.Adam(1e-3)
pl2_pi_opt = tf.keras.optimizers.Adam(3e-4)
pl2_bl_opt = tf.keras.optimizers.Adam(1e-3)

def train_baseline(bl_model, states, returns, bl_opt, epochs=4, batch_size=64):
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


def log_prob_of(pi_model, states, actions):
    probs = pi_model(states)
    logp_all = tf.math.log(probs + 1e-8)
    chosen = tf.reduce_sum(logp_all * tf.one_hot(actions, probs.shape[-1]), axis=1)
    return chosen, probs, logp_all

def train_actor(pi_model, states, actions, old_logps, advantages, pi_opt,
                    entropy_coeff, clip=0.2, epochs=4, batch_size=256, target_kl=0.02):
    
    n = states.shape[0]
    actions = tf.constant(actions, dtype=tf.int32)
    old_logps = tf.constant(old_logps, dtype=tf.float32)
    kl, loss = 0.0, 0.0
    
    for _ in range(epochs):
        idx = tf.random.shuffle(tf.range(n))
        
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            
            with tf.GradientTape() as tape:
                new_logp, probs, logp_all = log_prob_of(pi_model, tf.gather(states, b), tf.gather(actions, b))
                ratio = tf.exp(new_logp - tf.gather(old_logps, b))
                adv = tf.gather(advantages, b)
                surr = tf.minimum(ratio * adv, tf.clip_by_value(ratio, 1 - clip, 1 + clip) * adv)
                entropy = -tf.reduce_mean(tf.reduce_sum(probs * logp_all, axis=-1))
                loss = -tf.reduce_mean(surr) - entropy_coeff * entropy
            
            grads = tape.gradient(loss, pi_model.trainable_variables)
            pi_opt.apply_gradients(zip(grads, pi_model.trainable_variables))
        
        # early stop if the policy has moved too far from the data-collecting policy
        new_logp_chosen, _, _ = log_prob_of(pi_model, states, actions)
        kl = float(tf.reduce_mean(old_logps - new_logp_chosen))
        if kl > 1.5 * target_kl:
            break

    return loss, kl

def update_player(pi_model, bl_model, pi_opt, bl_opt, epochs, 
                  states, bl_states, actions, old_logps, returns, entropy_coeff):
    
    bl_values = tf.squeeze(bl_model(bl_states), axis=-1)
    ev_before = explained_variance(returns, bl_values)

    adv = returns - bl_values
    adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

    #lp, _, _ = log_prob_of(pi_model, states, tf.constant(actions, dtype=tf.int32))
    #print(float(tf.reduce_max(tf.abs(tf.exp(lp - old_logps) - 1))))

    pi_loss, kl = train_actor(pi_model, states, actions, old_logps, adv,
                                  pi_opt, entropy_coeff, epochs=epochs)
    bl_loss = train_baseline(bl_model, bl_states, returns, bl_opt, epochs=epochs)
    return ev_before, pi_loss, bl_loss, kl

def select_action(pi_model, state):
    
    state = tf.expand_dims(state, axis=0)
    action_probs = pi_model(state)
    action_probs = tf.squeeze(action_probs, axis=0) 
    logits = tf.math.log(action_probs + 1e-8)
    action = tf.random.categorical(logits[tf.newaxis, :], num_samples=1)
    action = int(action[0, 0])

    #print("action taken: ", action)
    
    return action, float(logits[action])

def explained_variance(pl_returns_ten, bl_value):
    return 1 - tf.math.reduce_variance(pl_returns_ten - bl_value) / tf.math.reduce_variance(pl_returns_ten)

def compute_returns(rewards, discount_rate):
    
    returns = []
    cm_rew = 0   # discounted cumulative return

    for r in reversed(rewards):
        cm_rew = r + discount_rate * cm_rew
        returns.insert(0, cm_rew)

    return tf.cast(tf.constant(returns), dtype=tf.float32)


states, pl1_actions, pl2_actions, pl1_rewards, pl2_rewards, = [], [], [], [], []
pl1_returns, pl2_returns, pl1_log_probs, pl2_log_probs = [], [], [], []

flip, active1, active2 = True, True, True

episode_len: int = 400
num_updates: int = 160
epoch_batch: int = 4

for epo in range(num_updates):

    sole_hunt_p1, sole_hunt_p2, hunt, plant_p1, plant_p2 = 0, 0, 0, 0, 0
    
    #try scaling the mauling punishment to not discourage early exploration:
    progress = epo / num_updates
    mauling_punishment = -1#-1.0 - (4.0 * progress)
    stag_reward = 5#10.0 - (5.0 * progress)

    times = []

    for i in range(epoch_batch):

        env = gym.make("StagHunt-Hunt-v0", obs_type='coords', stag_follows = False, stag_reward =stag_reward ,enable_multiagent = True, flip_obs = False, max_timesteps=10000, run_away_after_maul=True, mauling_punishment=mauling_punishment)  # you can pass config parameters here
        #initialization step
        obs, info = env.reset()

        old_rew = [0]*3
        old_rew[0], old_rew[1], old_rew[2] = (distances(obs[0]))

        state = one_hot_encoder(obs[0], entities, 5, 5)
        

        for epi in range(episode_len):    

            times.append(epi/episode_len)
            state = one_hot_encoder(obs[0], entities=entities)
            states.append(state)
            

            pl1_action,  pl1_log_prob = select_action(pl1_pi_model, state)
            pl2_action,  pl2_log_prob = select_action(pl2_pi_model, state)

            obs, reward, done, trunc, info = env.step([pl1_action, pl2_action])
            #env.render()
            #obs is a tensor, rewards is a tuple info is a dict, done and trunc are bool

            if(done or trunc):
                print("max number of moves reached")
                exit(1)


            pl1_actions.append(pl1_action)
            pl1_log_probs.append(pl1_log_prob)
            pl2_actions.append(pl2_action)
            pl2_log_probs.append(pl2_log_prob)

            current_rew = [0]*3
            current_rew[0], current_rew[1], current_rew[2] = distances(obs[0])
            shaping_factor = -(0.05*((gamma*current_rew[0] - old_rew[0])+(gamma*current_rew[1] - old_rew[1])+(gamma*current_rew[2] - old_rew[2])))
            old_rew = current_rew

            if(isinstance(reward, tuple)):

                #print("distance of 1: ", dist_1, " distance of 2: ", dist_2, " max dist: ", shaping_factor)

                pl1_rewards.append(reward[0] + shaping_factor)
                pl2_rewards.append(reward[1] + shaping_factor)

                if(reward[0]== 1): plant_p1 += 1
                if(reward[0]== mauling_punishment): sole_hunt_p1 += 1
                if(reward[0]== stag_reward): hunt += 1
                if(reward[1]== mauling_punishment): sole_hunt_p2 += 1
                if(reward[1]== 1): plant_p2 += 1

            else: 
                print("rewards error")
                exit(1)

            #obs, rewards, done, trunc, info = env.step([env.action_space.sample(), env.action_space.sample()])
            #env.render()
        
        #if(i == 1): sanity_check = compute_returns(pl1_rewards, 0.99)
        pl1_returns.append(compute_returns(pl1_rewards, gamma))
        pl2_returns.append(compute_returns(pl2_rewards, gamma))

        pl1_rewards.clear()
        pl2_rewards.clear()
                    
        env.close()

    stacked_states = tf.stack(states)
    times_tensor = tf.expand_dims(tf.constant(times), axis=-1)
    bl_states = tf.keras.ops.concatenate([stacked_states, times_tensor], axis=-1)

    #print("stacked states shape:", stacked_states.shape)
    #for j in range(len(sanity_check)):
    #    if(not(sanity_check[j] == pl1_returns[-1][j])): print("error: ", sanity_check[j], " vs ", pl1_returns[-1][j])
    
    pl_returns_ten = tf.keras.ops.concatenate(pl1_returns, axis =-1)
    #bl_values = tf.squeeze(pl1_bl_model(bl_states), axis =-1)    
    #print("explained variance for ",epo ," epoch and player 1 before training:" ,explained_variance(pl_returns_ten=pl_returns_ten, bl_value=bl_values))
    
    entropy_coeff = 1e-3#(5e-3)/(1+hunt)

    ev1, pl1, bl1, kl1 = update_player(pi_model=pl1_pi_model, bl_model=pl1_bl_model, pi_opt=pl1_pi_opt,
                bl_opt=pl1_bl_opt, states= stacked_states, bl_states=bl_states, 
                actions=pl1_actions, old_logps=pl1_log_probs, returns=pl_returns_ten, 
                entropy_coeff=entropy_coeff, epochs= epoch_batch)
    #print("for epoch: ", epo, " for player 1 we have a policy loss: ", pi_loss, " and a baseline loss: ", bl_loss)
    

    pl_returns_ten = tf.keras.ops.concatenate(pl2_returns, axis =-1)
    #bl_values = tf.squeeze(pl2_bl_model(bl_states), axis =-1)    
    #print("explained variance for ",epo ," epoch and player 2 before training:" ,explained_variance(pl_returns_ten=pl_returns_ten, bl_value=bl_values))
    
    ev2, pl2, bl2, kl2 = update_player(pi_model=pl2_pi_model, bl_model=pl2_bl_model, pi_opt=pl2_pi_opt,
                    bl_opt=pl2_bl_opt, states= stacked_states, bl_states=bl_states, 
                    actions=pl2_actions, old_logps=pl2_log_probs, returns=pl_returns_ten, 
                    entropy_coeff=entropy_coeff, epochs=epoch_batch)
    #print("for epoch: ", epo, " for player 2 we have a policy loss: ", pi_loss, " and a baseline loss: ", bl_loss)
    
    print("epoch ", epo, " | EV: " , ev1 , ev2, " | KL: ", kl1, kl2)
    print("number of times player 1 harvested plants: ", plant_p1, " number of times player 2 harvested plants:", plant_p2)
    print("number of successful hunts: ", hunt, " number of unsuccessful hunts for player 1: ", sole_hunt_p1, " and for player 2: ", sole_hunt_p2)
    
    pl_returns_ten = 0

    states.clear()
    stacked_states = tf.keras.ops.empty((2, 2))
    pl1_actions.clear()
    pl2_actions.clear()
    pl1_returns.clear()
    pl2_returns.clear()
    pl1_log_probs.clear()
    pl2_log_probs.clear()

    print("")

#plt.show()