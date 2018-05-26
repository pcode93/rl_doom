from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from env import initialize_vizdoom
from env.util import all_actions
from network.ac_policy import ActorCriticPolicy
from network.cnn import CNN
from network.q_network import QNetwork
from trajectory import TrajectoryGenerator


def test(show_game=True):
    path = 'results/basic_final'
    map_name = 'simpler_basic.cfg'
    game = initialize_vizdoom(map_name, show_game)
    actions = all_actions(game)

    policy = QNetwork(CNN(12), len(actions))

    agent = DQNAgent(policy, policy, None, None, None, cuda=True)
    generator = TrajectoryGenerator(map_name, game, 0, 0, agent, actions)

    generator.load(path)
    mean, std, max, min = generator.test(10)


if __name__ == '__main__':
    test()
