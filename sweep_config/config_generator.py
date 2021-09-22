import sys
sys.path.append("sweep_config")

from sweep_config_random import sweep_config_random
from sweep_config_grid import sweep_config_grid
from sweep_config_bayes import sweep_config_bayes

common_params = {
    "SAVEFIG_NAME"  : {
        "value":"result.png"
    },
    "NOW" : {
        "value" : "None"
    },
    "NUM_WORKERS":{
        "value" : 1
    },
    "DATASET":{
        "value": "MNIST-handwrite"
    },
    "DATA_PATH":{
        "value" : '../dataset/'
    },
    "ARCHITECTURE":{
        "value":"single-layer-perceptron",
    }
}


def get_sweep_config(sweep_config):
    
    if sweep_config.lower() == "random":
        config = sweep_config_random
    elif sweep_config.lower() == "grid":
        config = sweep_config_grid
    elif sweep_config.lower() == "bayes":
        config = sweep_config_bayes
    else:
        raise ValueError(
            "Invalid sweep config. It must be the one of [random, grid, bayes]"
        )
    
    config["parameters"].update(common_params)

    return config
