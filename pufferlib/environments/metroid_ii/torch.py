from functools import partial
import torch.nn as nn

import pufferlib.models


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy,
            input_size=512, hidden_size=512, num_layers=1):
        super().__init__(env, policy,
            input_size, hidden_size, num_layers)

class Policy(nn.Module):
    # flat_size and framestack were ambiguous choices
    def __init__(self, env, *args, framestack=1, flat_size=1280,
            input_size=512, hidden_size=512, output_size=512,
            channels_last=True, downsample=1, **kwargs):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)

        # The actual network
        self.network= nn.Sequential(
            
            # Had to shrink this so that it would fit, since input space is so
            # small
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=2)),
            nn.ReLU(),
            # was 4 kernel size
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor = pufferlib.pytorch.layer_init( nn.Linear(hidden_size, env.single_action_space.n), std=0.01)

        # critic
        self.value_fn = pufferlib.pytorch.layer_init( nn.Linear(output_size, 1), std=1)

    def forward(self, observations):
        hidden, lookup = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, lookup)
        return actions, value

    def encode_observations(self, observations):
        # print(f"self dtype: {self.dtype}")
        # input("hold")
        # observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, ::self.downsample, ::self.downsample]

        return self.network(observations.float() / 255.0), None

    def decode_actions(self, flat_hidden, lookup, concat=None):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value
