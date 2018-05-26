import torch
from collections import deque

import numpy as np


class Monitor:
    def post_game(self, score): pass

    def post_epoch(self, t, params): pass

    def pre_epoch(self, t): pass

    def done(self, params): pass


class ProgressMonitor(Monitor):
    def __init__(self, score_buffer_size, monitor_interval=10):
        self.score_buffer_size = score_buffer_size
        self.monitor_interval = monitor_interval
        self.scores = deque(maxlen=score_buffer_size)
        self.epoch_scores = []
        self.schedules = []

    def post_game(self, score):
        self.scores.append(score)

    def post_epoch(self, t, epoch_params):
        self.schedules.append({sched.name: sched.param for sched in epoch_params['schedules']})

        if (t + 1) % self.monitor_interval == 0:
            epoch_score = np.array(list(self.scores))
            self.epoch_scores.append(epoch_score)

            print("\EPOCH %d\n-------" % (t + 1))

            print("PARAMS:")
            for sched in epoch_params['schedules']:
                print("%s : %f" % (sched.name, sched.param))

            print("Scores: max: %.1f, min: %.1f, mean: %.1f +/- %.1f," % (epoch_score.max(), epoch_score.min(),
                                                                          epoch_score.mean(), epoch_score.std()))
            print('=================================================================================')


class CheckpointMonitor(Monitor):
    def __init__(self, env_params, agent):
        self.env_params = env_params
        self.agent = agent

    def save(self, t, progress_params):
        path = "{0}_{1}".format(self.env_params["save_path"], t)

        progress_params = {"loss": progress_params["loss"]}
        torch.save(self.env_params, path + "_env_params")
        torch.save(progress_params, path + "_progress_params")
        torch.save(self.agent.state(), path + "_agent")

    def post_epoch(self, t, progress_params):
        if (t + 1) % self.env_params["save_interval"] == 0:
            self.save(t, progress_params)

    def done(self, params):
        self.save("final", params)

    @staticmethod
    def load(path):
        env_params = torch.load(path + "_env_params")
        progress_params = torch.load(path + "_progress_params")
        agent_params = torch.load(path + "_agent")

        return env_params, progress_params, agent_params
