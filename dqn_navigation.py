from unityagents import UnityEnvironment
from dqn_agent import DQN_agent
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

FILEPATH = 'dqn_navigation.pt'
IS_TRAINING = False

env = UnityEnvironment(file_name="Banana.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=IS_TRAINING)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

nodesList = [64, 64, 64]

agent = DQN_agent(state_size, action_size, nodesList)

NUM_EPISODES = 800
EPISON_START = 1.0
EPISON_END = 0.01
EPISON_DECAY = 0.995 

eps = EPISON_START

if not IS_TRAINING:
    eps = 0.0
    agent.load_model(FILEPATH)

scores = []
scores_window = deque(maxlen=100)

for i_episode in range(1, NUM_EPISODES + 1):
    env_info = env.reset(train_mode=IS_TRAINING)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state, eps, IS_TRAINING)        # select an action
        
        env_info = env.step(action.astype(int))[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        
        if IS_TRAINING:
            agent.step(state, action, reward, next_state, done) #step agent
        
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step

        if done:                                       # exit loop if episode finished
            break

    print("Score: {}".format(score))

    scores_window.append(score)
    scores.append(score)

    if IS_TRAINING:
        eps = max(EPISON_END, EPISON_DECAY * eps)

    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

env.close()

if IS_TRAINING:
    agent.save_model(FILEPATH)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

