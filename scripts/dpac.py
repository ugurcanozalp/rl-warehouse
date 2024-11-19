
from argparse import ArgumentParser

from rlwarehouse.algos import DPAC

parser = ArgumentParser()
parser = DPAC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = DPAC(**dict_args)

agent.experiment()