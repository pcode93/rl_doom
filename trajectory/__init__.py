import pickle
import time
from collections import deque

import numpy as np
from tqdm import trange

from env.util import zero_frame, preprocess_frame, LazyFrames


class TrajectoryGenerator:
    def __init__(self, map_name, env, n_epochs, epoch_len, agent, actions,
                 frame_skip=4, frames_per_state=4, state_dim=(3, 60, 108), save_interval=None, out_name=None,
                 param_schedules=None, monitor=None, next_epoch_callback=None, shape_reward_fn=None):

        self.map_name = map_name
        self.env = env
        self.epochs = n_epochs
        self.epoch_len = epoch_len
        self.agent = agent
        self.param_schedules = param_schedules or []
        self.monitor = monitor
        self.epoch_cb = next_epoch_callback
        self.shape_reward_fn = shape_reward_fn
        self.frames_per_state = frames_per_state
        self.frame_skip = frame_skip
        self.state_dim = state_dim
        self.actions = actions
        self.save_interval = save_interval
        self.out_name = out_name

    def _frame_buffer(self):
        return deque([zero_frame(self.state_dim)
                      for _ in range(self.frames_per_state)], maxlen=self.frames_per_state)

    def _save(self, t):
        if not self.out_name: return

        params = {
            "map_name": self.map_name,
            "actions": self.actions,
            "frame_skip": self.frame_skip,
            "frames_per_state": self.frames_per_state,
            "state_dim": self.state_dim,
            "monitor": self.monitor
        }

        with open("{0}_{1}_params".format(self.out_name, t), 'wb') as output:
            pickle.dump(params, output)

        self.agent.save("{0}_{1}_model".format(self.out_name, t))

    def run(self):
        self.env.new_episode()

        for epoch in range(self.epochs):
            last_states = self._frame_buffer()

            if self.epoch_cb:
                self.epoch_cb(epoch)

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

                    if self.monitor:
                        self.monitor.game_finished(score)

                    self.env.new_episode()
                    state_vars = self.env.get_state().game_variables
                    prev_state_vars = state_vars

                    screen = preprocess_frame(self.env.get_state().screen_buffer, self.state_dim)
                    last_states.append(screen)

            loss = self.agent.epoch_finished()

            if self.monitor:
                self.monitor.epoch_finished(epoch, loss, self.param_schedules)

            if self.save_interval and epoch % self.save_interval == 0:
                self._save(epoch + 1)

        self._save("final")
        self.env.close()

    def load(self, path):
        with open(path + '_params', 'rb') as input:
            params = pickle.load(input)

        for key, val in params.items():
            setattr(self, key, val)

        self.agent.load(path + "_model")

    def test(self, n_games):
        scores = []

        for t in trange(n_games, leave=False):
            self.env.new_episode()
            last_states = self._frame_buffer()

            while True:
                screen = preprocess_frame(self.env.get_state().screen_buffer, self.state_dim)
                last_states.append(screen)

                action = self.agent.act(t, LazyFrames(list(last_states)))
                self.env.make_action(self.actions[action], self.frame_skip)

                time.sleep(0.075)

                if self.env.is_episode_finished():
                    scores.append(self.env.get_total_reward())
                    break

        scores = np.array(scores)
        return scores.mean(), scores.std(), scores.max(), scores.min()
