from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from ddpg_agent import Agent
import matplotlib.pyplot as plt
import torch

env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")


def ddpg(reward_objective=0.5):
    """DDPG.

    Params
    ======
        reward_objective (int): the reward that should be reached for 100 consecutive episodes for the agent to be trained
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    episodes_over_objective = 0
    i_episode = 0
    t=0
    scores = []
    scores_window = deque(maxlen=100)  # A queue to keep only the last 100 episodes' scores

    while episodes_over_objective < 100:
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)  # initialize the score (for each agent)

        # agent.reset()

        while True:  # A loop for the iterations
            actions = agent.act(states)  # select an action (for each agent)
            actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            #print("actions")
            #print(actions)
            env_info = env.step(actions)[brain_name]  # send all actions to the environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            for i in range(num_agents):
                agent.step(states[i], actions[i], rewards[i] * 0.1, next_states[i], dones[i], t)
            score += rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            t+=1
            if np.any(dones):  # exit loop if episode finished
                break

        scores_window.append(np.max(score))  # save most recent score
        scores.append(np.max(score))  # save most recent score
        print('\rEpisode {}\tMaximum Score: {:.2f}'.format(i_episode, np.max(score)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= reward_objective:
            episodes_over_objective += 1
        else:
            episodes_over_objective = 0

        if episodes_over_objective == 100:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.actor_local.state_dict(), 'actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_checkpoint.pth')

            break
        i_episode += 1
    return scores


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
num_agents = len(env_info.agents)
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
states = env_info.vector_observations
print('States look like:', states)
state_size = states.shape[1]
print('States have length:', state_size)

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

scores = ddpg()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Maximum Score')
plt.xlabel('Episode #')
plt.savefig("Scores.png")

env.close()
