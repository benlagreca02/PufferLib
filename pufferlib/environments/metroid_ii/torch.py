from functools import partial
import torch
import torch.nn as nn

import pufferlib.models
from metroid_env import *


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy,
            input_size=1024, hidden_size=1024, num_layers=1):
        super().__init__(env, policy,
            input_size, hidden_size, num_layers)

class Policy(nn.Module):
    # This is all horribly hacked
    def __init__(self, env, *args, framestack=1, flat_size=(4036),
            input_size=512, hidden_size=1024, output_size=512,  # input_size doesn't really get used for my code oops
            channels_last=True, downsample=1, **kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)

        # The network for feature extraction from screen data
        self.screen_network= nn.Sequential( 
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=3)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # The network for encoding all features, three layers cause why not
        self.feature_network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )


        self.actor = pufferlib.pytorch.layer_init( nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        # critic
        self.value_fn = pufferlib.pytorch.layer_init( nn.Linear(output_size, 1), std=1)

    # Compute the forward pass of the networks (actor and critic)
    # Broken into encode, and decode for LSTM wrapper
    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        # Nativize the tensor
        observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)

        # Pop out the screen observations, we need to extract features first
        screen_obs = observations.pop(SCREEN_OBS)

        if self.channels_last:
            screen_obs = screen_obs.permute(0, 3, 1, 2)
        # Downsample happens in environment itsself in my case
        # Could and should probably remove this
        if self.downsample > 1:
            screen_obs = screen_obs[:, :, ::self.downsample, ::self.downsample]

        # Run CNN on screen to extract features
        screen_features = self.screen_network(screen_obs.float()/255.0)

        # print(f"SCREEN FEATURES DIMS: {screen_features.shape}")
        # Get features from the dict
        health = observations[HEALTH_OBS] / 100.0
        missiles = observations[MISSILE_OBS]
        upgrades = observations[MAJOR_UPGRADES_OBS]
        beam = observations[BEAM_OBS]

        # print("SHAPES: ")
        # print(f"screen: {screen_features.shape}")
        # print(f"health: {health.shape}")
        # print(f"missiles: {missiles.shape}")
 
        all_features = torch.cat((screen_features, health, missiles, upgrades, beam), dim=1)
        # print(f"all: {all_features.shape}")
        hidden = self.feature_network(all_features)
       
        # Something to do with entity observations
        lookup=None
        return hidden, lookup

        # return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
