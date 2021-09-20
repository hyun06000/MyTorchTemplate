import sys
sys.path.append("sweep_config")

from sweep_config_random import sweep_config_random
from sweep_config_grid import sweep_config_grid
from sweep_config_bayes import sweep_config_bayes


def get_sweep_config(sweep_config):
    if sweep_config.lower() == "random":
        return sweep_config_random
    elif sweep_config.lower() == "grid":
        return sweep_config_grid
    elif sweep_config.lower() == "bayes":
        return sweep_config_bayes
    else:
        raise ValueError(
            "Invalid sweep config. It must be the one of [random, grid, bayes]"
        )