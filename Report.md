# Report

## Implementation
In this implementation, we use DDPG to train the agents. Since each agent receives its own local observation, both agents use the same actor and critic networks, and their experiences are used to train the agents, which speeds up learning.
The DDPG main and target networks for the actor and critic are implemented using PyTorch. 

The neural networks for the actor and critic comprised a batch normalization layer after the input layer, and two hidden layers of 256 and 128 units respectively for the actor network, and two hidden layers of 256 units each for the critic network, with ReLu activation functions, and the activation function of the last layer for the actor being the hyperbolic tangent function (tanh) to get an output between 1 and -1. For each action, the Ornstein Uhlenbeck process is employed to generate noise that is added to encourage exploration.

The solution implements experience replay, to break correlation between the experiences, and thus avoid bias during learning, the experiences are stored in a buffer of a size of 100000, and randomly sampled in batches of 128.

The neural network weights are updated using the ADAM optimizer, with a learning rate of 1e-4 for the actor, and 1e-3 for the critic, and a discount factor gamma of 0.95. Finally, the learning is performed 5 times using different samples every 4 steps, and the target networks are updated with a soft update using a tau value of 1e-3.

Note that to stabilize learning, the rewards returned by the environment are multiplied by a factor of 0.1. 
## Results
The model was trained on a GPU, and was able to reach an average reward over the 20 agents of over 30 after 127 episodes, the average reward remained over 30 for a 100 consecutive episodes as shown in the figure below. The environment was solved after 227 episodes, with an average reward of 36.56.

![alt text](https://github.com/nassimatoumi/MADDPG-Tennis/blob/4f466eff2b97a3b4b06751796513891bff44578b/Scores.png)

The saved weights can be found in the files **critic_checkpoint.pth** and **actor_checkpoint.pth** 

## Future works
Although the agents were able to reach the target score, multiple enhancements might be considered for future works to improve the performance:
- Prioritized Experience Replay, which should allow the model to converge more quickly by reusing experiences that the model learns from the most (biggest TD error). 

- Other methods such as TD3 (Twin Delayed DDPG) could be explored for a more stable learning.

- Deeper neural networks might also improve the obtained results.
