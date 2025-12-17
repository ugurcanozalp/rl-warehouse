
from argparse import ArgumentParser

from rlwarehouse.algos import QDSAC

parser = ArgumentParser()
parser = QDSAC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = QDSAC(**dict_args)

agent.experiment()