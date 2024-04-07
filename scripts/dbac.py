
from argparse import ArgumentParser

from rlwarehouse.algos import DBAC

parser = ArgumentParser()
parser = DBAC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = DBAC(**dict_args)

agent.experiment()