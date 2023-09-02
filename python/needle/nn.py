"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(
            init.kaiming_uniform(
                self.out_features, 1, device=device, dtype=dtype, requires_grad=bias).reshape((1, self.out_features)))

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight
        if self.bias.requires_grad:
            return y + self.bias.broadcast_to(y.shape)
        else:
            return y


class Flatten(Module):
    def forward(self, x):
        batch_size = x.shape[0]
        dim = 1
        for i in range(1, len(x.shape)):
            dim = dim * x.shape[i]
        return ops.reshape(x, (batch_size, dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        y = ops.relu(x)
        return y


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        y = x
        for module in self.modules:
            y = module.forward(y)
        return y


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        num_classes = logits.shape[1]
        y_one_hot = init.one_hot(num_classes, y)
        losses = ops.logsumexp(logits, axes=(1,)) - ops.summation(logits * y_one_hot, axes=(1,))
        return ops.summation(losses / losses.shape[0])


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))

        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        assert (self.dim == x.shape[1])

        if self.training:
            batch_size = x.shape[0]

            # Calculate mean and variance for this batch
            batch_mean = ops.summation(x, axes=(0,)) / batch_size
            batch_var = ops.summation((x - ops.broadcast_to(batch_mean, x.shape)) ** 2, axes=(0,)) / batch_size

            # Update running average of mean/variance
            self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + batch_var * self.momentum

            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        return ops.broadcast_to(self.weight, x.shape) * (
                (x - ops.broadcast_to(mean, x.shape)) /
                (ops.broadcast_to(var, x.shape) + self.eps) ** 0.5) + ops.broadcast_to(self.bias, x.shape)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))

    def forward(self, x: Tensor) -> Tensor:
        assert (self.dim == x.shape[1])
        batch_size = x.shape[0]
        mean = ops.broadcast_to(ops.reshape(ops.summation(x, axes=(1,)) / self.dim, (batch_size, 1)), x.shape)
        variance = ops.broadcast_to(ops.reshape(ops.summation((x - mean) ** 2, axes=(1,)) / self.dim, (batch_size, 1)),
                                    x.shape)
        return ops.broadcast_to(self.weight, x.shape) * ((x - mean) / (variance + self.eps) ** 0.5) + ops.broadcast_to(
            self.bias, x.shape)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
