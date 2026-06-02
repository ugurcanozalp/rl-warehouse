
from argparse import ArgumentParser

from rlwarehouse.algos import STLQN

parser = ArgumentParser()
parser = STLQN.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = STLQN(**dict_args)

agent.experiment()