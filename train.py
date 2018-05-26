import torch
import argparse
from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from env import initialize_vizdoom
from env.util import all_actions, default_actions_for_map
from experience_replay import ReplayMemory
from monitor import ProgressMonitor, CheckpointMonitor
from network.ac_policy import ActorCriticPolicy
from network.capsnet import CapsNet
from network.cnn import CNN
from network.q_network import QNetwork
from trajectory import TrajectoryGenerator
from util import LinearSchedule, LRWrapper


def train():
    parser = argparse.ArgumentParser(description='Train an agent in the ViZDoom environment.')
    parser.add_argument('map_name', help='path to the map config')
    parser.add_argument('--output_path', dest='output_path', help='output path for agent checkpoints')
    parser.add_argument('--save_interval', dest='save_interval', default=10, type=int,
                        help='interval, measured in epochs, between each agent checkpoint')
    parser.add_argument('--cuda', dest='cuda', default=True, action='store_true', help='whether to use cuda')
    parser.add_argument('--log_interval', dest='log_interval', default=10, type=int,
                        help='interval between each progress update log')
    parser.add_argument('--score_buffer_size', dest='score_buffer_size', default=50, type=int,
                        help='the amount of last scores that will be saved to compute statistics')

    parser.add_argument('--n_epochs', dest='n_epochs', default=1000, type=int, help='number of epochs')
    parser.add_argument('--epoch_len', dest='epoch_len', default=1024, type=int, help='the length of an epoch')
    parser.add_argument('--lr', dest='lr', default=2.5e-4, type=float, help='learning rate')
    parser.add_argument('--lr_decay', dest='decay_lr', default=True, help='whether to decay learning rate each epoch')
    parser.add_argument('--gamma', dest='gamma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--batch_size', dest='batch_size', default=32, type=float, help='batch size')
    parser.add_argument('--alg', dest='alg', default='ppo', choices=['ppo', 'dqn'],
                        help='the algorithm the agent will use')
    parser.add_argument('--nn', dest='nn', default='deepmind_cnn', choices=['deepmind_cnn', 'capsnet'],
                        help='neural network that the agent will use as its feature network')

    parser.add_argument('--frame_skip', dest='frame_skip', default=4, type=int,
                        help='number of frames to skip each action')
    parser.add_argument('--frames_per_state', dest='frames_per_state', default=4, type=int,
                        help='number of frames to stack every state')
    parser.add_argument('--state_w', dest='state_w', default=108, type=int,
                        help='target state width to resize each frame to')
    parser.add_argument('--state_h', dest='state_h', default=60, type=int,
                        help='target state height to resize each frame to')
    parser.add_argument('--state_rgb', dest='rgb', default=True, action='store_true',
                        help='whether to use rgb or gray frames')
    parser.add_argument('--shape_reward', dest='shape_reward', default=True, action='store_true',
                        help='whether to shape rewards')

    parser.add_argument('--ppo_n_timesteps', dest='n_timesteps', default=1024, type=int,
                        help='number of timesteps for each PPO iteration')
    parser.add_argument('--ppo_lambda', dest='lam', default=0.95, type=float, help='lambda value for GAE')
    parser.add_argument('--ppo_eps', dest='eps', default=0.1, type=float, help='clipping parameter for PPO')
    parser.add_argument('--ppo_decay_params', dest='ppo_decay', default=True, action='store_true',
                        help='whether to decay PPO learning rate and epsilon each epoch linearly')
    parser.add_argument('--ppo_ent_coeff', dest='ent_coeff', default=0.01, type=float,
                        help='entropy coefficient for PPO')
    parser.add_argument('--ppo_value_coeff', dest='value_coeff', default=1.0, type=float,
                        help='value coefficient for PPO')
    parser.add_argument('--ppo_opt_epochs', dest='opt_epochs', default=4, type=int,
                        help='number of optimization epochs for PPO')

    parser.add_argument('--dqn_use_ddqn', dest='ddqn', default=True, action='store_true',
                        help='whether to use ddqn instead of dqn')
    parser.add_argument('--dqn_dueling', dest='dueling', default=True, action='store_true',
                        help='whether to use a dueling architecture in dqn')
    parser.add_argument('--dqn_min_eps', dest='min_eps', default=0.01, type=float,
                        help='minimum value of epsilon for dqn')
    parser.add_argument('--dqn_mem_size', dest='memory_size', default=100000, type=int,
                        help='replay memory size for dqn')
    parser.add_argument('--dqn_init_size', dest='init_size', default=10000, type=int,
                        help='number of timesteps before dqn starts learning')
    parser.add_argument('--dqn_q_update_interval', dest='q_update_interval', default=1, type=int,
                        help='the interval between updates of the q function')
    parser.add_argument('--dqn_target_update_interval', dest='target_update_interval', default=1000, type=int,
                        help='the interval between updated of the target q function')

    args = parser.parse_args()

    game = initialize_vizdoom(args.map_name)
    #actions = all_actions(game)
    actions = default_actions_for_map(game, args.map_name)

    in_channels = args.frames_per_state * (3 if args.rgb else 1)

    if args.nn == 'deepmind_cnn':
        feature_net = CNN(in_channels)
    elif args.nn == 'capsnet':
        feature_net = CapsNet(in_channels)

    if args.alg == 'ppo':
        policy = ActorCriticPolicy(feature_net, len(actions))
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

        eps_sched = LinearSchedule("eps", args.eps, 1, args.n_epochs, end_val=1.0 if not args.ppo_decay else 0.0)
        lr_sched = LRWrapper(optimizer, LinearSchedule("lr", args.lr, 1, args.n_epochs, end_val=1.0 if not args.ppo_decay else 0.0))
        schedules = [lr_sched, eps_sched]

        agent = PPOAgent(policy, optimizer, eps_sched, cuda=args.cuda, n_timesteps=args.n_timesteps,
                         batch_size=args.batch_size, opt_epochs=args.opt_epochs, gamma=args.gamma, lam=args.lam,
                         entropy_coeff=args.ent_coeff, value_coeff=args.value_coeff)
    elif args.alg == 'dqn':
        q = QNetwork(feature_net, len(actions))
        tq = QNetwork(feature_net, len(actions))
        optimizer = torch.optim.Adam(q.parameters(), lr=args.lr)

        memory = ReplayMemory(args.memory_size)
        eps_sched = LinearSchedule("eps", 1, 1, args.n_epochs, end_val=args.min_eps)
        lr_sched = LRWrapper(optimizer, LinearSchedule("lr", args.lr, 1, args.n_epochs, end_val=1.0 if not args.decay_lr else 0.0))
        schedules = [lr_sched, eps_sched]

        agent = DQNAgent(q, tq, optimizer, memory, eps_sched, cuda=args.cuda, init_steps=args.init_size,
                         q_update_interval=args.q_update_interval, target_update_interval=args.target_update_interval,
                         ddqn=args.ddqn, gamma=args.gamma, batch_size=args.batch_size)

    progress_monitor = ProgressMonitor(args.score_buffer_size, monitor_interval=args.log_interval)

    env_params = {
        "env": {
            "frame_skip": args.frame_skip,
            "frames_per_state": args.frames_per_state,
            "state_dim": (3 if args.rgb else 1, args.state_h, args.state_w),
            "actions": actions
        },
        "agent": {
            "alg": args.alg,
            "nn": args.nn
        },
        "save_path": args.output_path,
        "save_interval": args.save_interval,
        "progress_monitor": progress_monitor,
        "map_name": args.map_name
    }

    if args.output_path:
        checkpoint_monitor = CheckpointMonitor(env_params, agent)
        monitors = [checkpoint_monitor, progress_monitor]
    else:
        monitors = [progress_monitor]

    generator = TrajectoryGenerator(game, args.n_epochs, args.epoch_len if args.alg != 'ppo' else args.n_timesteps,
                                    agent, monitors=monitors, param_schedules=schedules, **env_params["env"])

    generator.run()


if __name__ == '__main__':
    train()
