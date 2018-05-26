import torch
from torch.autograd import Variable


def var(x, cuda=True):
    v = Variable(x).type(torch.FloatTensor)

    if cuda:
        v = v.cuda()

    return v


def np_var(x, cuda=True):
    return var(torch.from_numpy(x), cuda)


class ParamSchedule:
    def __init__(self, name, param_start, schedule_func):
        self.param_start = param_start
        self.param = param_start
        self.func = schedule_func
        self.name = name

    def step(self):
        self.param = self.param_start * self.func(self.param_start)


class LRWrapper:
    def __init__(self, lr_sched):
        self.lr = lr_sched
        self.name = "lr"

    def step(self): self.lr.step()

    @property
    def param(self): return self.lr.optimizer.param_groups[0]["lr"]


class LinearSchedule:
    def __init__(self, start_val, max_iters, start_at=0, min_val=0.0):
        self._start_val = start_val
        self._val = start_val
        self._max_iters = max_iters
        self._min = min_val
        self._start = start_at

    def step(self, t):
        if t >= self._start:
            self._val = max(self._min, self._start_val - ((t - self._start) / self._max_iters))

    def val(self, *args): return self._val
