
import math

import torch as th
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus


LOG2 = math.log(2.)

class StableTanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.

    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    This is more stable alternative to `TanhTransform`.

    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _call(self, x):
        finfo = th.finfo(x.dtype)
        return x.tanh().clamp(-1+finfo.eps, 1-finfo.eps)

    def _inverse(self, y):
        # We clamp to the boundary here.
        finfo = th.finfo(y.dtype)
        return th.atanh(y.clamp(-1+finfo.eps, 1-finfo.eps))

    def log_abs_det_jacobian(self, x, y):
        finfo = th.finfo(y.dtype)
        return (1 - y.clamp(-1+finfo.eps, 1-finfo.eps)**2).log()
        # return (1 - y*y + finfo.eps**2).log() # + 2*finfo.eps


class BoundingTanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = lb+(ub-lb)*\sigma(x)`.

    It is equivalent to
    ```
    ComposeTransform([SigmoidTransform(), AffineTransform(lb, (ub-lb))])
    ```
    This is stable bounded distribution using `tanh` function with given lower and upper bound.

    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.

    """
    bijective = True
    sign = +1

    def __init__(self, lb, ub, cache_size=0):
        super().__init__(cache_size=cache_size)
        self._lb = lb
        self._ub = ub 
        self._span = ub - lb
        self._scale = (ub-lb)/2
        self._bias = (ub+lb)/2

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.real

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.interval(self._lb, self._ub)

    def __eq__(self, other):
        return isinstance(other, BoundingTanhTransform)

    def _call(self, x):
        finfo = th.finfo(x.dtype)
        return self._bias + self._scale*x.tanh().clamp(finfo.eps, 1-finfo.eps)

    def _inverse(self, y):
        # We clamp to the boundary here.
        finfo = th.finfo(y.dtype)
        yy = (y-self._bias)/self._scale
        return th.atanh(yy.clamp(-1+finfo.eps, 1-finfo.eps))
    
    def log_abs_det_jacobian(self, x, y):
        finfo = th.finfo(y.dtype)
        yy = ((y-self._lb)/self._span)
        return (self._scale*(1-yy.clamp(-1+finfo.eps, 1-finfo.eps)**2)).log()
    
    # dy/dx = span * sigmoid(x)*(1-sigmoid(x))
    # sigmoid(x) = (y-lb)/span
    # ==> dy/dx = span * (y-lb)/span * (1-(y-lb)/span)
    # ==> dy/dx = (y-lb) * (span-y+lb)/span = (y-lb) * (ub-y) / span

    # y = bias + scale * tanh(x) ==> tanh(x) = (y-bias)/scale == yy
    # dy/dx = scale * (1-tanh(x)^2) = scale * (1 - yy^2)
