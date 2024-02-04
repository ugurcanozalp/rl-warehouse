
from .policy import ContinuousMLPPolicy
from .value import ContinuousMLPValue
from .qvalue import ContinuousMLPQValue
from .probabilistic_qvalue import ContinuousMLPStochasticQValue
from .probabilistic_value import ContinuousMLPStochasticValue
from .model import ContinuousMLPModel
from .reward import ContinuousMLPReward

policy_map = {
    "continuous_mlp2": ContinuousMLPPolicy
}

value_map = {
    "continuous_mlp2": ContinuousMLPValue
}

qvalue_map = {
    "continuous_mlp2": ContinuousMLPQValue
}

probabilistic_qvalue_map = {
    "continuous_mlp2": ContinuousMLPStochasticQValue
}

model_map = {
    "continuous_mlp2": ContinuousMLPModel
}

reward_map = {
    "continuous_mlp2": ContinuousMLPReward
}
