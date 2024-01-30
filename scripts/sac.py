
from argparse import ArgumentParser
import pytorch_lightning as pl 

from rlwarehouse.algos import SAC

parser = ArgumentParser()
parser = SAC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = SAC(**dict_args)

agent.experiment()