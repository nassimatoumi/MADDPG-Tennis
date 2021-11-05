# MADDPG-Tennis
[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "dlrnd kernel"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

- After each episode, we add up the rewards that each agent received, to get a final score for each agent. We then take the maximum of these 2 scores, which is considered the episode's score.

The environment is considered solved, once the average of those **scores** is at least +0.5 for more than 100 episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in this folder, and unzip (or decompress) the file. 
3. Depending on the operating system, the path in line 8 of both the **train_agent** and **use_agent** files should be set as follows:
    - Mac: "Tennis.app"
    - Windows (x86): "Tennis_Windows_x86/Tennis.exe"
    - Windows (x86_64): "Tennis_Windows_x86_64/Tennis.exe"
    - Linux (x86): "Tennis_Linux/Tennis.x86"
    - Linux (x86_64): "Tennis_Linux/Tennis.x86_64"

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

## Instructions
- To train the agent from scratch, run the following command, while specifying in the file the path of the newly created models of the actor and critic lines 61 and 62 respectively if needed:

    **python3 train_agent.py**

Note that the hyperparameter values can be tuned in the ddpg_agent file, and the Deep Neural Network architecture can be changed in the file model.py 

- To use the trained model, run the following command:

    **python3 use_agent.py**
    
The path of the model file of the actor and critic can be specified in lines 38 and 39 respectively. 
