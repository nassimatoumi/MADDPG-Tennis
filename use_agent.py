from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from ddpg_agent import Agent
import matplotlib.pyplot as plt
import torch

env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")
num_episodes = 10

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
states = env_info.vector_observations
print('States look like:', states)
state_size = states.shape[1]
print('States have length:', state_size)

agent = Agent(state_size, action_size, 0)
#Load the model
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

agent.actor_local.load_state_dict(torch.load('actor_checkpoint.pth',map_location=map_location))
agent.critic_local.load_state_dict(torch.load('critic_checkpoint.pth',map_location=map_location))

for i in range(num_episodes):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    score = np.zeros(len(env_info.agents))
    while True:
        actions = agent.act(states)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get the next state
        rewards = env_info.rewards  # get the reward
        dones = env_info.local_done  # see if episode has finished
        states = next_states
        score += rewards
        if np.any(dones):  # exit loop if episode finished
            break
    print("Episode "+str(i))
    print("Maximum score "+str(np.max(score)))
    print('\rEpisode {}\tMaximum Score: {:.2f}'.format(i, np.max(score)), end="")

env.close()
