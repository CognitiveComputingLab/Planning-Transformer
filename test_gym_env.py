import gym
import d4rl
import numpy as np
from models import DT
env_name = "antmaze-medium-diverse-v2"
traj,info =  DT.load_d4rl_trajectories(env_name)
# print(info['obs_mean'][0][:3])
# print(info['obs_std'][0][:3])
#
# obs = np.concatenate([np.array(seq['observations']) for seq in traj])
# print(np.max(obs[:, :3],axis=0))
# print(np.min(obs[:, :3],axis=0))
#
goals = np.concatenate([np.array(seq['goals']) for seq in traj])
print(np.max(goals[:, :3],axis=0))
print(np.min(goals[:, :3],axis=0))
#
# for i in range(100):
#     print(len(traj[i]['goals']),len(traj[i]["observations"]))
#     print(tuple(traj[i]['goals'][-1]),tuple(traj[i]['observations'][0, :2]), tuple(traj[i]['observations'][-1, :2]))

starts = np.array([traj_i['observations'][0][:3] for traj_i in traj])
print(np.max(starts, axis=0))
print(np.min(starts, axis=0))