
from argparse import ArgumentParser

from rlwarehouse.algos import REDQ

parser = ArgumentParser()
parser = REDQ.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = REDQ(**dict_args)

agent.experiment()