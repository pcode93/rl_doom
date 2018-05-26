import itertools

import torch
import skimage.transform
import numpy as np

from vizdoom.vizdoom import Button


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out


def encode_actions(game, actions):
    def enc(btns):
        encoded = [0] * n_btns

        for btn in btns:
            encoded[buttons[btn]] = 1

        return encoded

    n_btns = game.get_available_buttons_size()
    buttons = {btn: i for i, btn in enumerate(game.get_available_buttons())}

    return [enc(btns) for btns in actions]


def all_actions(game):
    return [list(action) for action in itertools.product([0, 1], repeat=game.get_available_buttons_size())]


def preprocess_frame(frame, dim):
    return skimage.transform.resize(frame, dim).astype(np.float32)


def zero_frame(dim):
    return np.zeros(dim)


def stack_frames(xs):
    return torch.from_numpy(np.vstack([np.expand_dims(np.array(x), axis=0) for x in xs]))


def default_reward_shaping(map_name):
    if map_name == 'defend_the_center.cfg':
        def shape_reward(reward, state_vars, prev_state_vars):
            if state_vars[0] < prev_state_vars[0]:
                reward -= 0.175

            if state_vars[1] < prev_state_vars[1]:
                reward -= - 0.1

            return reward
    elif map_name == 'deathmatch.cfg':
        def shape_reward(reward, state_vars, prev_state_vars):
            if state_vars[0] > prev_state_vars[0]:
                reward += 10

            if state_vars[0] < prev_state_vars[0]:
                reward -= 10

            if state_vars[1] < prev_state_vars[1]:
                reward -= 0.1 * (prev_state_vars[1] - state_vars[1])

            if state_vars[1] > prev_state_vars[1]:
                reward += 0.09 * (state_vars[1] - prev_state_vars[1])

            if state_vars[2] < prev_state_vars[2]:
                reward -= 0.05 * (prev_state_vars[2] - state_vars[2])

            if state_vars[2] > prev_state_vars[2]:
                reward += 0.04 * (state_vars[2] - prev_state_vars[2])

            if state_vars[4] < prev_state_vars[4]:
                reward -= 0.05 * (prev_state_vars[4] - state_vars[4])

            if state_vars[4] > prev_state_vars[4]:
                reward += 0.04 * state_vars[4] - prev_state_vars[4]

            return reward
    else:
        def shape_reward(reward, *args):
            return reward

    return shape_reward


def default_actions_for_map(game, map_name):
    if map_name == 'health_gathering.cfg':
        actions = [
            [Button.TURN_RIGHT],
            [Button.TURN_RIGHT, Button.MOVE_FORWARD],
            [Button.MOVE_FORWARD]
        ]

    return encode_actions(game, actions)
