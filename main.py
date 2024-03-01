# main.py
from lightning.pytorch.cli import LightningCLI
from Algorithms import BehaviouralCloning, ActorCritic
from Tasks import TwentyQuestions

def cli_main():
    cli = LightningCLI(save_config_kwargs={"overwrite": True})

if __name__ == "__main__":
    cli_main()