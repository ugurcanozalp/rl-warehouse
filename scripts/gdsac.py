
from argparse import ArgumentParser

from rlwarehouse.algos import GDSAC

parser = ArgumentParser()
parser = GDSAC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = GDSAC(**dict_args)

agent.experiment()