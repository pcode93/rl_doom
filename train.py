import torch

from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from env import initialize_vizdoom
from env.util import all_actions
from experience_replay import ReplayMemory
from monitor import Monitor
from network.ac_policy import ActorCriticPolicy
from network.cnn import CNN
from network.q_network import QNetwork
from trajectory import TrajectoryGenerator
from util import ParamSchedule, LinearSchedule, LRWrapper


def train2():
    map_name = 'simpler_basic.cfg'
    game = initialize_vizdoom(map_name)
    actions = all_actions(game)

    lr = 1e-4
    n_epochs = 50

    q = QNetwork(CNN(12), len(actions))
    tq = QNetwork(CNN(12), len(actions))
    optimizer = torch.optim.Adam(q.parameters(), lr=lr)

    memory = ReplayMemory(10000)
    alpha_sched = LinearSchedule(1, n_epochs)
    #lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha_sched.val)
    eps_sched = ParamSchedule("eps", 1, alpha_sched.val)

    agent = DQNAgent(q, tq, optimizer, memory, eps_sched, cuda=True)
    monitor = Monitor(10)
    generator = TrajectoryGenerator(map_name, game, n_epochs, 1000, agent, actions, next_epoch_callback=alpha_sched.step,
                                    out_name="results/basic", monitor=monitor, param_schedules=[eps_sched])

    generator.run()


def train():
    map_name = 'simpler_basic.cfg'
    game = initialize_vizdoom(map_name)
    actions = all_actions(game)

    lr = 2.5e-4
    n_epochs = 100

    policy = ActorCriticPolicy(CNN(12), len(actions))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    alpha_sched = LinearSchedule(1, n_epochs, min_val=1.0)
    lr_sched = LRWrapper(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=alpha_sched.val))
    eps_sched = ParamSchedule("eps", 1, alpha_sched.val)

    agent = PPOAgent(policy, optimizer, eps_sched, cuda=True)
    monitor = Monitor(10)
    generator = TrajectoryGenerator(map_name, game, n_epochs, 1024, agent, actions, next_epoch_callback=alpha_sched.step,
                                    out_name="results/basic", monitor=monitor, param_schedules=[lr_sched, eps_sched])

    generator.run()


if __name__ == '__main__':
    train2()
