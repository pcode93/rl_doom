import os
from vizdoom import *

_dir_path = os.path.dirname(os.path.abspath(__file__))


def initialize_vizdoom(map_name, visible=False):
    game = DoomGame()
    game.load_config(os.path.join(_dir_path, "../maps/" + map_name))
    game.set_window_visible(visible)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.RGB24)
    game.set_screen_resolution(ScreenResolution.RES_400X225)
    game.init()

    return game
