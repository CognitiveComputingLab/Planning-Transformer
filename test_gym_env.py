import gym
import d4rl
import numpy as np
env_name = "antmaze-medium-diverse-v2"
dataset =  gym.make(env_name).get_dataset()
print(np.concatenate((dataset['infos/goal'][:400,:2], dataset['infos/qpos'][:400, :2], dataset['observations'][:400, :2], dataset['rewards'][:400, np.newaxis]),axis=1))