
from argparse import ArgumentParser

from rlwarehouse.algos import ESTAC

parser = ArgumentParser()
parser = ESTAC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = ESTAC(**dict_args)

agent.experiment()