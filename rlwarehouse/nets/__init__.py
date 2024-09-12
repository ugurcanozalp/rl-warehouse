
from .policy import ContinuousMLPPolicy
from .probabilistic_policy import ContinuousMLPStochasticPolicy
from .value import ContinuousMLPValue
from .probabilistic_value import ContinuousMLPStochasticValue
from .qvalue import ContinuousMLPQValue
from .probabilistic_qvalue import ContinuousMLPStochasticQValue
from .quantile_qvalue import ContinuousMLPQuantileQValue
from .model import ContinuousMLPModel
from .reward import ContinuousMLPReward
from .probabilistic_reward import ContinuousMLPStochasticReward

policy_map = {
    "continuous_mlp2": ContinuousMLPPolicy
}

probabilistic_policy_map = {
    "continuous_mlp2": ContinuousMLPStochasticPolicy
}

value_map = {
    "continuous_mlp2": ContinuousMLPValue
}

probabilistic_value_map = {
    "continuous_mlp2": ContinuousMLPStochasticValue
}

qvalue_map = {
    "continuous_mlp2": ContinuousMLPQValue
}

probabilistic_qvalue_map = {
    "continuous_mlp2": ContinuousMLPStochasticQValue
}

quantile_qvalue_map = {
    "continuous_mlp2": ContinuousMLPQuantileQValue
}

model_map = {
    "continuous_mlp2": ContinuousMLPModel
}

reward_map = {
    "continuous_mlp2": ContinuousMLPReward
}

probabilistic_reward_map = {
    "continuous_mlp2": ContinuousMLPStochasticReward
}