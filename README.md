![Trained Agent](checkpoints/trained_model.gif)

# Project 2: 
This is my solution for the mutli-agent version of Project 2 - Continuous Control of Udacity's Deep Reinforcement Learning nanodegree.

### Project Description
The goal is to control a double-jointed robtic arm so that its 'hand' is in constant contact with a moving target zone. For every frame that the arm touches the target zone, a reward of 0.1 is given.\
As input, the arm accepts 4 values between -1 and 1 where each value represents the amount of torque that should be applied to one of its four joints. The obervation space consists of 33 variables that describe the current position, rotation, velocity and angular velocity of the arm.  \
The project is considered solved, if average score over 100 episodes exceeds 30. In case of mulitple agents, the score if an episode is considered the average over all agents during that episode. (I deviated slightly from this criteria, which is described in the 'train.py' section in REPORT.md)

### Setup
The setup for this repo is almost identical to https://github.com/udacity/deep-reinforcement-learning#dependencies so make sure to follow the installation instructions there.  \
For training, the multi-agent environment is expected to be located at `<git root>/Reacher_Linux_multi`
For testing, the single-agent environment is expected to be located at `<git root>/Reacher_Linux_single`

If you wish to run with one of the pretrained models from the checkpoint folder, torch must be upgraded to a newer version:
``` bash
pip3 install --upgrade torch
```

Furthermore, there is a Dockerfile in this repo with which it is possible to utilize AMD GPUs for training. To use it, make sure docker is installed, then simply execute the `startDocker.sh` script. This will download and start the image.

### Training
In order to train the model, adjust the hyperparameters in `agent.py` to your liking, then simply execute
```
python3 train.py

### Testing
To test your trained models, you can execute `test.py`. The script expects the folder in which the model files a stored as input. Eg:
``` bash
python3 test.py checkpoint
```

### Result
During the first 10 to 20 epochs of training, the agent achieved almost exponential progress and reached the target value of 30 after epoch 20. It continued to improve its score up until around epoch 30 to a value of around 37 to 39 where it stayed for the remainder of the training. \
Therefore it is safe to say, that the agent solved the environment after 20 epochs.

![progress](checkpoints/progress.png)


### Implementation Details
I solved the problem using the Deep Deterministic Policy Gradients (DDPG) algorithm.
See the [report](REPORT.md) for a detailed description of the implementation and design choices.

### Artifacts
Trained models are stored in the checkpoints folder, together with an image of their training progress as seen below.

