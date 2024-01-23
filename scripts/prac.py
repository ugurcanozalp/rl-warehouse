
from argparse import ArgumentParser
import pytorch_lightning as pl 

from rlwarehouse.algos import PRAC

parser = ArgumentParser()
parser = PRAC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = PRAC(**dict_args)

agent.train()