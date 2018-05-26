import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def squash(x, dim):
    norm = x.norm(dim=dim, keepdim=True)
    return (norm ** 2 / (1 + norm ** 2)) * x / norm


class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, num_caps, caps_dim, kernel_size, stride):
        super(PrimaryCapsule, self).__init__()

        self.num_caps = num_caps
        self.caps_dim = caps_dim

        self.conv = nn.Conv2d(in_channels, num_caps * caps_dim, kernel_size, stride)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1, self.caps_dim)

        return squash(x, dim=-1)


class FinalCapsule(nn.Module):
    def __init__(self, n_iters, num_caps_in, num_caps_out, caps_dim_in, caps_dim_out):
        super(FinalCapsule, self).__init__()

        self.n_iters = n_iters
        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out
        self.caps_dim_in = caps_dim_in
        self.caps_dim_out = caps_dim_out

        self.W = nn.Parameter(torch.randn(num_caps_in, num_caps_out, caps_dim_out, caps_dim_in))

    def forward(self, x):
        xh = torch.matmul(self.W, x[:, :, None, :, None]).squeeze(-1)
        b = Variable(torch.zeros(x.shape[0], self.num_caps_in, self.num_caps_out)).cuda()
        xhd = xh.detach()

        for i in range(self.n_iters - 1):
            c = F.softmax(b, dim=2)
            s = (c.unsqueeze(-1) * xhd).sum(dim=1, keepdim=True)
            v = squash(s, dim=-1)
            b = b + (xhd * v).sum(-1)

        c = F.softmax(b, dim=2)
        s = (c.unsqueeze(-1) * xh).sum(dim=1, keepdim=True)
        v = squash(s, dim=-1)

        return v.squeeze(1)


class CapsNet(nn.Module):
    def __init__(self, in_channels):
        super(CapsNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=6, stride=3)
        self.pc = PrimaryCapsule(32, 32, 8, 3, 2)
        self.dc = FinalCapsule(3, 32 * 4 * 6, 64, 8, 16)
        self.out_dim = 1024

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pc(x)
        x = self.dc(x)

        return x.view(x.shape[0], self.out_dim)
