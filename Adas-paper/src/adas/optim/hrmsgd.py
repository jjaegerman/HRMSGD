"""
"""
from torch.optim.optimizer import Optimizer, required

import sys

import numpy as np
import torch

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .hmetrics import Metrics
else:
    from optim.hmetrics import Metrics


class HRMSGD(Optimizer):
    """
    Vectorized SGD from torch.optim.SGD
    """

    def __init__(self,
                 params,
                 listed_params,
                 saves,
                 MAX = 4000,
                 S = 100,
                 measure = "SQLRF",
                 lr: float = required,
                 beta: float = 0, #LR momentum
                 zeta: float = 1, #LR dampening
                 momentum: float = 0, #SGD momentum
                 dampening: float = 0, #SGD dampening
                 weight_decay: float = 0,
                 nesterov: bool = False
                 ):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(HRMSGD, self).__init__(params, defaults)

        # Adas Specific stuff (not SGD)
        if np.less(beta, 0) or np.greater_equal(beta, 1):
            raise ValueError(f'Invalid beta: {beta}')
        self.beta = beta
        self.metrics = metrics = Metrics(params=listed_params, MAX=MAX, S=S, measure=measure)
        self.lr_vector = np.repeat(a=lr, repeats=len(metrics.params))
        self.init_lr = lr
        self.zeta = zeta
        self.saves = saves
        self.start = 0

    def __setstate__(self, state):
        super(HRMSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def batch_update(self, batch):
        self.metrics()
        if(self.saves[batch]):
            measures = self.metrics.update()
            self.lr_vector = self.lr_vector*self.beta + self.zeta*measures


    def step(self, closure: callable = None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        iteration_group = 0
        for group in self.param_groups:
            iteration_group += 1
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p_index, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(
                            d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                p.data.add_(d_p, alpha=-self.lr_vector[p_index])

        return loss
