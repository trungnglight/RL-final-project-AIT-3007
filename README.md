# MAgent2 RL Final Project
## Overview
In this final project, you will develop and train a reinforcement learning (RL) agent using the MAgent2 platform. The task is to solve a specified MAgent2 environment, and your trained agent will be evaluated on all following three types of opponents:

1. Random Agents: Agents that take random actions in the environment.
2. A Pretrained Agent: A pretrained agent provided in the repository.
3. A Final Agent: A stronger pretrained agent, which will be released in the final week of the course before the deadline.

Your agent's performance should be evaluated based on reward and win rate against each of these models. 

<p align="center">
  <div style="display: flex; justify-content: center; gap: 10px;">
    <video width="45%" controls>
      <source src="video/random.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <video width="45%" controls>
      <source src="video/pretrained.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</p>

## Installation
clone this repo and install with
```
pip install -r requirements.txt
```

## Demos
See `main.py` for a starter code.

## References

1. [MAgent2 GitHub Repository](https://github.com/Farama-Foundation/PettingZoo/tree/master)
2. [MAgent2 API Documentation](https://magent2.farama.org/introduction/basic_usage/)

For further details on environment setup and agent interactions, please refer to the MAgent2 documentation.