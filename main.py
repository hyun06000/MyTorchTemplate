
import wandb

from sweep_config import sweep_config
from trainer.trainer import train

def main():

    wandb.login()
    sweep_id = wandb.sweep(
    project="simple-classification-sweep-2",
    sweep = sweep_config
    )

    count = 5 # number of runs to execute
    wandb.agent(sweep_id, function=train, count=count)

if __name__ == "__main__":
    main()