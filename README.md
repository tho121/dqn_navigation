# Project 1: Nagivation with Deep Q-Network by Tony Ho

### Project Environment Details

This project is about training an agent to nagivate an environment that creates yellow and blue bananas.  By moving the agent to a banana, the agent will recieve a reward of +1 or -1.
After a certain time period, the environment is reset.

For this project, the environment is considered solved when the last 100 episodes average a score of at least 13.0

The observational space is of type continuous with a size of 37  
The action space is of type discrete with a size of 4

### Getting Started

Make sure to install the packages Unity ML-Agents, NumPy, PyTorch (v0.4) and Matlibplot  
Also, install Unity3D with the Linux Build Support option enabled

In the file 'dqn_navigation.py', change NUM_EPISODES to a low number like 10.  Then, from the command line, type in 'python ./PROJECT_PATH/dqn_navigation.py'  
By default, this file is set to load the trained agent.  You can watch the agent navigate the environment and the score is printed in the command line window.  After every 100 episodes, the average score is printed.  

The best way to check the average score after 100 episodes for the trained agent is to change NO_GRAPHICS to True and NUM_EPISODES to 100, then run it.

You can also test the training model by changing IS_TRAINING to True and I recommend changing the FILEPATH to something else.  I've solved the environment at 500 episodes, but I've set it to 800 to be safe, so set NUM_EPISODES to 800, then run it.