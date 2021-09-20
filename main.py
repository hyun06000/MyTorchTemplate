
import wandb

from sweep_config.sweep_config_bayes import sweep_config_bayes
from trainer.trainer import train

def main():

    wandb.login()
    sweep_id = wandb.sweep(
    project="simple-classification-sweep-bayes",
    sweep = sweep_config_bayes
    )

    count = 5 # number of runs to execute
    wandb.agent(sweep_id, function=train, count=count)

if __name__ == "__main__":
    main()