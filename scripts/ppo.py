
from argparse import ArgumentParser
import pytorch_lightning as pl 

from rlwarehouse.algos import PPO

parser = ArgumentParser()
parser = PPO.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = PPO(**dict_args)

agent.train()