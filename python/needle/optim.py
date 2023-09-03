"""Optimization module"""
import needle as ndl
import numpy as np
import needle.init as init


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param not in self.u:
                self.u[param] = init.zeros(*param.shape, device=param.device, dtype=param.dtype, requires_grad=False)
            self.u[param].data = self.momentum * self.u[param].data + (1 - self.momentum) * (
                    param.grad.data + self.weight_decay * param.data)
            param.data = param.data - self.lr * self.u[param].data


class Adam(Optimizer):
    def __init__(
            self,
            params,
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t = self.t + 1
        for param in self.params:
            if param not in self.m:
                self.m[param] = init.zeros(*param.shape, device=param.device, dtype=param.dtype, requires_grad=False)
            if param not in self.v:
                self.v[param] = init.zeros(*param.shape, device=param.device, dtype=param.dtype, requires_grad=False)

            self.m[param].data = self.beta1 * self.m[param].data + (1 - self.beta1) * (
                    param.grad.data + self.weight_decay * param.data)
            self.v[param].data = self.beta2 * self.v[param].data + (1 - self.beta2) * (
                    param.grad.data + self.weight_decay * param.data) ** 2

            m = self.m[param].data / (1 - self.beta1 ** self.t)
            v = self.v[param].data / (1 - self.beta2 ** self.t)

            param.data = param.data - self.lr * m / (v ** 0.5 + self.eps)
