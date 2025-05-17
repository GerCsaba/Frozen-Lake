import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training, render):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None) # Frozen Lake environment

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # initializare tabelul q
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    
    learning_rate_a = 0.9 #learning rate (α) (0-1) 1- înseamnă că agentul învață mai repede, dar poate să nu rețină bine cunoștințele anterioare
                                            # 0-înseamnă că agentul se bazează mai mult pe cunoștințele anterioare, învățând lent din experiențele noi.
    discount_factor_g = 0.9 # gamma or discount rate (γ) (0-1). 1-agentul acorde prioritate rewardurile pe termen lung
                                                            # 0-agentul acorde prioritate recompenselor imediate 
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # Cat de repede va scadea epsilonul
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes) # array goala pentru recompesna
    

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up exploreaza, genereaza actiuni random
                
            else:
                action = np.argmax(q[state,:]) # daca nu exploreaza atunci alege un actiune din tabelul q 

            new_state,reward,terminated,truncated,_ = env.step(action) # executa actiunea,  actualizează va valorile 
            print(reward)
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )
            # print(q[:])

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

       

    env.close()

    # Vizualizarea/salvarea graficonul
    sum_rewards = np.zeros(episodes)
    if is_training:
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.savefig('frozen_lake8x8-2.png')

    if is_training:
        f = open("frozen_lake8x8-2.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    # run(15000)
    run(30, is_training=True, render=False)
    
