from pdb import set_trace as T

import gymnasium
from gymnasium.wrappers import TimeLimit
import functools

# From my Metroid-II-RL repo, after calling `pip install -e .` in said repo
from metroid_env import MetroidEnv

import pufferlib.emulation
import pufferlib.postprocess

# makes partial environment creation function
def env_creator(name='metroid_ii'):
    return functools.partial(make, name)

def make(name, render_mode='rgb_array', buf=None):
    '''Metroid II'''
    speed = 1 if render_mode == 'human' else 0
    # If we are renderingit as a human, we probably want to watch it

    # From Metroid-II-RL repo, that was installed with `pip install -e .`
    env = MetroidEnv(render_mode=render_mode, emulation_speed_factor=speed)
    env = TimeLimit(env, max_episode_steps=MetroidEnv.DEFAULT_EPISODE_LENGTH)

    env = pufferlib.postprocess.EpisodeStats(env)
    # Should be much faster than using my old way of doing it
    # Can't use this! My observation space includes other things aswell!
    # env = pufferlib.postprocess.ResizeObservation(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
