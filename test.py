import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from env import initialize_vizdoom
from monitor import CheckpointMonitor
from network.ac_policy import ActorCriticPolicy
from network.capsnet import CapsNet
from network.cnn import CNN
from network.q_network import QNetwork
from trajectory import TrajectoryGenerator


def save_recording(ims, name):
    fig = plt.figure()
    ims = [[plt.imshow(im)] for im in ims]

    ani = animation.ArtistAnimation(fig, ims, interval=75, blit=True, repeat_delay=1000)
    ani.save(name + '.mp4')


def test():
    parser = argparse.ArgumentParser(description='Test an agent in the ViZDoom environment.')
    parser.add_argument('agent_path', help='path to the agent checkpoint')
    parser.add_argument('--show_game', dest='show_game', default=False, action='store_true',
                        help='whether to show the game while agent is playing')
    parser.add_argument('--record', dest='record', default=False, action='store_true',
                        help='whether to record the agent playing')
    parser.add_argument('--output_path', dest='output_path', help='output path for the replay')
    parser.add_argument('--cuda', dest='cuda', default=False, action='store_true', help='whether to use cuda')
    parser.add_argument('--n_games', dest='n_games', default=1, type=int, help='number of games to play')

    args = parser.parse_args()
    env_params, progress_params, agent_params = CheckpointMonitor.load(args.agent_path)

    game = initialize_vizdoom(env_params["map_name"], args.show_game)
    actions = env_params["env"]["actions"]

    in_channels = env_params["env"]["state_dim"][0] * env_params["env"]["frames_per_state"]
    if env_params["agent"]["nn"] == 'deepmind_cnn':
        feature_net = CNN(in_channels)
    elif env_params["agent"]["nn"] == 'capsnet':
        feature_net = CapsNet(in_channels)

    if env_params["agent"]["alg"] == 'ppo':
        policy = ActorCriticPolicy(feature_net, len(actions))
        agent = PPOAgent(policy, None, None, cuda=args.cuda)
    elif env_params["agent"]["alg"] == 'dqn':
        q_net = QNetwork(feature_net, len(actions))
        agent = DQNAgent(q_net, q_net, None, None, None, cuda=args.cuda)

    agent.load(agent_params)

    checkpoint_monitor = CheckpointMonitor(env_params, agent)
    generator = TrajectoryGenerator(game, 0, 0, agent, param_schedules=progress_params.get("schedules", None),
                                    monitors=[checkpoint_monitor, env_params["progress_monitor"]], **env_params["env"])

    mean, std, max, min, frames = generator.test(args.n_games, args.record)

    print("Score: %1.f +/- %1.f, max: %1.f, min: %1.f" % (mean, std, max, min))

    if args.record:
        save_recording(frames, args.output_path)


if __name__ == '__main__':
    test()
