
from argparse import ArgumentParser

from rlwarehouse.algos import TOPSAC

parser = ArgumentParser()
parser = TOPSAC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = TOPSAC(**dict_args)

agent.experiment()