
from .policy import ContinuousMLP2Policy
from .probabilistic_policy import ContinuousMLP2StochasticPolicy
from .value import ContinuousMLP2Value
from .probabilistic_value import ContinuousMLP2StochasticValue
from .qvalue import ContinuousMLP2QValue, ContinuousMLP3QValue
from .probabilistic_qvalue import ContinuousMLP2StochasticQValue, ContinuousMLP3StochasticQValue
from .quantile_qvalue import ContinuousMLP2QuantileQValue, ContinuousMLP3QuantileQValue
from .model import ContinuousMLP2Model
from .reward import ContinuousMLP2Reward
from .probabilistic_reward import ContinuousMLP2StochasticReward

policy_map = {
    "continuous_mlp2": ContinuousMLP2Policy
}

probabilistic_policy_map = {
    "continuous_mlp2": ContinuousMLP2StochasticPolicy
}

value_map = {
    "continuous_mlp2": ContinuousMLP2Value
}

probabilistic_value_map = {
    "continuous_mlp2": ContinuousMLP2StochasticValue
}

qvalue_map = {
    "continuous_mlp2": ContinuousMLP2QValue, 
    "continuous_mlp3": ContinuousMLP3QValue
}

probabilistic_qvalue_map = {
    "continuous_mlp2": ContinuousMLP2StochasticQValue, 
    "continuous_mlp3": ContinuousMLP3StochasticQValue
}

quantile_qvalue_map = {
    "continuous_mlp2": ContinuousMLP2QuantileQValue, 
    "continuous_mlp3": ContinuousMLP3QuantileQValue
}

model_map = {
    "continuous_mlp2": ContinuousMLP2Model
}

reward_map = {
    "continuous_mlp2": ContinuousMLP2Reward
}

probabilistic_reward_map = {
    "continuous_mlp2": ContinuousMLP2StochasticReward
}