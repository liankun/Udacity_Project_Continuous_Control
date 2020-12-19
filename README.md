[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 


### Instructions

The Project is done in windows 64bit environment. 
- Continuous_Control.ipynb : The main code to train and evalue the agent (one agent version)
- Continuous_Control-20.ipynb: The main code to train and evalue the agent (twenty agents version)
- Agent.py : the code of ReplayPool and Agent to implement DDPG
- Model.py : the structor of the actor and critic
- checkpoint_actor_gpu.pth : the trained actor weight
- checkpoint_critic_gpu.pth: the trained critic weight
- checkpoint_actor_gpu_20agents.pth : the trained actor weight
- checkpoint_critic_gpu_20agents.pth: the trained critic weight
- Report.ipynb : report of the project
- Reacher_Windows_x86_64_one : The unity environment for the project (one agent version)
- Reacher_Windows_x86_64_twenty : The unity environment for the project (twenty agents version)

Environment setting (windows):
- create virtual environment: conda create --name project_continuous_control python=3.6.9
- activate environment: conda activate project_continuous_control
- install neccessary packets:
  1. install pytorch: conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch 
  2. install unityagents: pip install unityagents
  3. install matplotlib: conda install matplotlib
- clone the repository: git clone https://github.com/liankun/Udacity_Project_Continuous_Control.git
- cd Udacity_Project_Navigation
- create IPython kernel: python -m ipykernel install --user --name project_navigation --display-name "project_navigation" <br/>

Now you can run the project by using the project_navigation environment in the jupyter notebook.