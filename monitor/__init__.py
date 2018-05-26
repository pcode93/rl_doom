from collections import deque

import numpy as np


class Monitor:
    def __init__(self, score_buffer_size, monitor_interval=10):
        self.score_buffer_size = score_buffer_size
        self.monitor_interval = monitor_interval
        self.scores = deque(maxlen=score_buffer_size)
        self.epoch_scores = []
        self.losses = []
        self.schedules = []

    def game_finished(self, score):
        self.scores.append(score)

    def epoch_finished(self, t, loss, schedules):
        self.losses.append(loss)
        self.schedules.append({sched.name: sched.param for sched in schedules})

        if (t + 1) % self.monitor_interval == 0:
            epoch_score = np.array(list(self.scores))
            self.epoch_scores.append(epoch_score)

            print("\EPOCH %d\n-------" % (t - 1))

            print("PARAMS:")
            for sched in schedules:
                print("%s : %f" % (sched.name, sched.param))

            print("Scores: max: %.1f, min: %.1f, mean: %.1f +/- %.1f," % (epoch_score.max(), epoch_score.min(),
                                                                           epoch_score.mean(), epoch_score.std()))
            print('=================================================================================')
