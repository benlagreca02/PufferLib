from pdb import set_trace as T

import gymnasium
import functools

# from pokegym import Environment
# From my Metroid-II-RL repo, after calling `pip install -e .`
from metroid_env import MetroidEnv

import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='metroid_ii'):
    return functools.partial(make, name)

def make(name, render_mode=None, buf=None):
    '''Metroid II'''
    # From Metroid-II-RL repo
    env = MetroidEnv(render_mode=render_mode)
    # env = RenderWrapper(env)
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

'''
class RenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        self.env = env

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()
'''
