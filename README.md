## Hierarchical Reinforcement Learning for Autonomous Cross-Country Soaring <img src="images/logo.png" align="right" width=100/>

### Overview
Autonomous soaring constitutes an appealing task for applying reinforcement learning
methods within the scope of guidance, navigation, and control for aerospace applications. 
Cross-country soaring embraces a threefold decision-making dilemma between covering distance,
exploiting updrafts, and mapping the environment. 

This repository includes a reinforcement learning framework for the competition task of *GPS Triangle*
racing with a remotely controlled glider aircraft. The framework was developed at the
[Institute of Flight Mechanics and Controls (iFR)](https://www.ifr.uni-stuttgart.de/) at the 
University of Stuttgart. The trained agents were successfully tested with two UAVs from the Institute.
More detailed information about the problem statement, the hierarchical reinforcement learning approach
and the flight test results can be found in our most recent paper:

> Hier könnte ihr Paper stehen! Idealerweise direkt mit Bibtex key


![Dummy image](images/example_image.png)

### Getting started
This repository contains the full code, which was used to train our agent. 
The *glider* training environment is an extension of the [OpenAI gym](https://gym.openai.com/) library. 
It uses a 3 degrees of freedom model (3 DoF) in the presence of wind to simulate the gilder movement.
 
#### Prerequisites
To run the training environment you need to install a virtual Python 3.8 environment with 
the following packages:
gym (0.17.1), 
pytorch (1.4),
numpy (1.12.3),
scipy (1.6.2),
pandas (1.1.3) and
matplotlib (3.4.3).

To register the gilder module in your virtual environment, navigate to the folder
 *<parent_folder>/rl_ccs_sources/* and run 
```
pip install -r glider
```

### Credits
If you like to use our work in an academic context please cite:
