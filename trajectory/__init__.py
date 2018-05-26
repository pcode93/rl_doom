import time
from collections import deque

import numpy as np
from tqdm import trange

from env.util import zero_frame, preprocess_frame, LazyFrames
from util import rgb_frame


class TrajectoryGenerator:
    def __init__(self, env, n_epochs, epoch_len, agent, actions,
                 frame_skip=4, frames_per_state=4, state_dim=(3, 60, 108),
                 param_schedules=None, monitors=None, shape_reward_fn=None):

        self.env = env
        self.epochs = n_epochs
        self.epoch_len = epoch_len
        self.agent = agent
        self.param_schedules = param_schedules or []
        self.monitors = monitors
        self.shape_reward_fn = shape_reward_fn
        self.frames_per_state = frames_per_state
        self.frame_skip = frame_skip
        self.state_dim = state_dim
        self.actions = actions

    def _frame_buffer(self):
        return deque([zero_frame(self.state_dim)
                      for _ in range(self.frames_per_state)], maxlen=self.frames_per_state)

    def run(self):
        self.env.new_episode()
        epoch_params = {}

        for epoch in range(self.epochs):
            last_states = self._frame_buffer()

            for monitor in self.monitors:
                monitor.pre_epoch(epoch)

            for schedule in self.param_schedules:
                schedule.step()

            state_vars = self.env.get_state().game_variables
            prev_state_vars = state_vars

            screen = preprocess_frame(self.env.get_state().screen_buffer, self.state_dim)
            last_states.append(screen)

            for t in trange(self.epoch_len, leave=False):
                state = LazyFrames(list(last_states))

                action = self.agent.act(t, state)
                reward = self.env.make_action(self.actions[action], self.frame_skip)
                terminal = self.env.is_episode_finished()

                next_g_state = self.env.get_state()

                if not terminal:
                    state_vars = next_g_state.game_variables

                if self.shape_reward_fn:
                    reward = self.shape_reward_fn(reward, state_vars, prev_state_vars)

                prev_state_vars = state_vars

                next_screen = preprocess_frame(next_g_state.screen_buffer, self.state_dim) \
                    if not terminal else zero_frame(self.state_dim)

                last_states.append(next_screen)
                next_state = LazyFrames(list(last_states))

                self.agent.post_act(t, reward, terminal, next_state)

                if self.env.is_episode_finished():
                    last_states = self._frame_buffer()
                    score = self.env.get_total_reward()

                    for monitor in self.monitors:
                        monitor.post_game(score)

                    self.env.new_episode()
                    state_vars = self.env.get_state().game_variables
                    prev_state_vars = state_vars

                    screen = preprocess_frame(self.env.get_state().screen_buffer, self.state_dim)
                    last_states.append(screen)

            loss = self.agent.epoch_finished()
            epoch_params = {
                "loss": loss,
                "schedules": self.param_schedules
            }

            for monitor in self.monitors:
                monitor.post_epoch(epoch, epoch_params)

        for monitor in self.monitors:
            monitor.done(epoch_params)

        self.env.close()

    def test(self, n_games, record=False):
        scores = []
        frames = []

        for t in trange(n_games, leave=False):
            self.env.new_episode()
            last_states = self._frame_buffer()

            while True:
                if record:
                    frames.append(rgb_frame(self.env.get_state().screen_buffer))

                screen = preprocess_frame(self.env.get_state().screen_buffer, self.state_dim)
                last_states.append(screen)

                action = self.agent.act(t, LazyFrames(list(last_states)))
                self.env.make_action(self.actions[action], self.frame_skip)

                time.sleep(0.075)

                if self.env.is_episode_finished():
                    scores.append(self.env.get_total_reward())
                    break

        scores = np.array(scores)
        return scores.mean(), scores.std(), scores.max(), scores.min(), frames
