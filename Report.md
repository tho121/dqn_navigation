# Project 1 Report: Navigation with Deep Q-Network by Tony Ho

### Learning Algorithm

The algorithm used to solve this environment was a Deep Q-Learning algorithm.  This is an extension of the Q-Learning algorithm, which at its core, maps environment states to reward values.
Machine learning is added to Q-Learning to improve correlation between continuous values and the rest of the values in the state.

Adding machine learning to the Q-Learning algorithm requires two additional features, called Experience Replay and Fixed Q-Targets.

Experience Replay stores states that have been experienced and then is put in a replay buffer.
The experiences are then randomly sampled from replay buffer and used to update the neural network.
This breaks any effect order has on the states as well as trains the neural network.

Fixed Q-Targets is used to stabilize the neural network when updating it.  Updating after every sample results in the trained values oscillating so the network always overshoots the optimal value.
By having a seperate network, one network can be used to update the other network after a set number of experiences gathered.
A local network is used to determine the best action and the resulting experiences are used to update a seperate 'target' network.
After a while, the local network is updated with the trained values of the target network, resulting in stable learning.

The hyperparameters used are as follows:

Number of episodes = 800
Minimum epsilon = 0.01
Epsilon decay rate = 0.995
Buffer size = 10000
Batch size = 64
Gamma = 0.99
TAU = 0.001
Learning rate = 0.0005
Network update interval = 4

The network used was 3 fully connected layers with 64 nodes each.

### Plot of Rewards

The agent usually achieves an average score of at least 13.0 by episode 500.

![Trained Agent](./Figure_1.png)

### Ideas for Future Work

This was a basic DQN, so there are many features that could be added to improve learning.
Algorithm improvements include using Prioritized Experience Replay or Dueling DQN.
The most effective algorithm seems to be call Rainbow, which both previously mentioned algorithms, as well as Multi-Step Bootstrap Targets,
Distributional DQN and Noisy DQN.  This combination of algorithms is shown to outperform any individual algorithm by a significant amount when used to play Atari 2600 games.
