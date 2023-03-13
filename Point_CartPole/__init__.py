# -*- coding: utf-8 -*-


from gym.envs.registration import register
from point_env import PointEnv
#from envs.point_env_for_distral import PointEnv

register(
        'Point-v0',
        entry_point='point_env:PointEnv',
        max_episode_steps=40,
        #kwargs={"env_num": 1}
        )
