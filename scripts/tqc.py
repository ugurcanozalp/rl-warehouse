
from argparse import ArgumentParser

from rlwarehouse.algos import TQC

parser = ArgumentParser()
parser = TQC.add_model_specific_args(parser)
args = parser.parse_args()
dict_args = vars(args)
agent = TQC(**dict_args)

agent.experiment()