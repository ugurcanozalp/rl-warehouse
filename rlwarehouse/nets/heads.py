
import torch as th
from torch import nn
from torch.distributions import Distribution, Normal, Gamma, Categorical, TransformedDistribution, MultivariateNormal 
from torch.distributions.transforms import TanhTransform, SigmoidTransform, AffineTransform, ComposeTransform
from .utils import StableTanhTransform


EPS=1e-6

class GaussianHead(nn.Module):
    def __init__(self, n):
        super(GaussianHead, self).__init__()
        self._n = n

    def forward(self, x):
        mean = x[...,:self._n]
        logstd = x[...,self._n:].clip(-10, None)
        std = th.nn.functional.softplus(logstd, beta=1.0)
        dist = Normal(mean, std, validate_args=False)
        return dist


class SquashedGaussianHead(nn.Module):
    def __init__(self, n):
        super(SquashedGaussianHead, self).__init__()
        self._n = n

    def forward(self, x):
        # bt means before tanh
        mean_bt = x[...,:self._n] 
        logstd_bt = x[...,self._n:].clip(-10, None)
        std_bt = th.nn.functional.softplus(logstd_bt, beta=1.0) # this shit is stable 
        dist_bt = Normal(mean_bt, std_bt, validate_args=False)
        transform = TanhTransform(cache_size=1)
        dist = TransformedDistribution(dist_bt, [transform], validate_args=False)
        return dist


class GammaHead(nn.Module):
    def __init__(self, n):
        super(GammaHead, self).__init__()
        self._n = n

    def forward(self, x):
        concentration = th.nn.functional.softplus(x[...,:self._n])
        rate = th.nn.functional.softplus(x[...,self._n:]) + 1e-6
        dist = Gamma(concentration, rate, validate_args=False)
        return dist


class CategoricalHead(nn.Module):
    def __init__(self, n):
        super(CategoricalHead, self).__init__()
        self._n = n

    def forward(self, x):
        logit = x
        probs = th.nn.functional.softmax(logit)
        dist = Categorical(probs, validate_args=False)
        return dist


class DeterministicHead(nn.Module):
    def __init__(self, n):
        super(DeterministicHead, self).__init__()
        self._n = n

    def forward(self, x):
        mean = x
        y = mean
        dist = None
        return y

