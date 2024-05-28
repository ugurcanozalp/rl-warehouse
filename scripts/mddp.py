
from argparse import ArgumentParser

from rlwarehouse.algos import MDDP

parser = ArgumentParser()
parser = MDDP.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = MDDP(**dict_args)

agent.experiment()