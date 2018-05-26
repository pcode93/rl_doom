import numpy as np
import torch
from torch.autograd import Variable


def var(x, cuda=True):
    v = Variable(x).type(torch.FloatTensor)

    if cuda:
        v = v.cuda()

    return v


def np_var(x, cuda=True):
    return var(torch.from_numpy(x), cuda)


def rgb_frame(frame):
    im = np.zeros_like(frame)

    im[:, :, 2] = frame[:, :, 0]
    im[:, :, 1] = frame[:, :, 1]
    im[:, :, 0] = frame[:, :, 2]

    return np.copy(im)


class LRWrapper:
    def __init__(self, optimizer, schedule):
        self.schedule = schedule
        self.name = self.schedule.name
        self.lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: self.schedule.decay_val)

    def step(self, t):
        self.schedule.step(t)
        self.lr.step()

    @property
    def param(self): return self.lr.optimizer.param_groups[0]["lr"]


class LinearSchedule:
    def __init__(self, name, init_param_val, start_val, max_iters, start_at=0, end_val=0.0):
        self.name = name
        self.init_param_val = init_param_val
        self.start_val = start_val
        self.decay_val = start_val
        self.param = init_param_val
        self.max_iters = max_iters
        self.end_val = end_val
        self.start = start_at

    def step(self, t):
        if t >= self.start:
            self.decay_val = max(self.end_val, self.start_val - ((t - self.start) / self.max_iters))
            self.param = self.decay_val * self.init_param_val
