import numpy as np


class ExponentialScheduler(object):
    max_steps: int
    decay_rate: float
    max_value: int

    def __init__(self, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.decay_rate = kwargs.get('decay_rate', 0.1)
        self.max_value = kwargs.get('max_value', 1.0)

    def __call__(self, step):
        return float(self.max_value / (1.0 + np.exp(-self.decay_rate * (step - self.max_steps))))


class ExponentialSchedulerGumbel(object):

    def __init__(self, **kwargs):
        self.min_tau = kwargs.get('min_tau')
        self.max_tau = kwargs.get('max_tau')
        self.decay_rate = kwargs.get('decay_rate')

    def __call__(self, step):
        t = np.maximum(self.max_tau * np.exp(-self.decay_rate * step), self.min_tau)
        return t


class LinearScheduler(object):
    def __init__(self, **kwargs):
        self.waiting_steps = kwargs.get('waiting_steps', 0)
        self.annealing_steps = kwargs.get('annealing_steps', 1000)

        self.start_value = kwargs.get('start_value', 1)
        self.end_value = kwargs.get('end_value', 0)

    def __call__(self, step):
        step = max(0, step - self.waiting_steps)
        step = min(step, self.annealing_steps)

        out = self.start_value + (self.end_value - self.start_value) * step / self.annealing_steps
        return out


class ConstantScheduler(object):
    def __init__(self, **kwargs):
        self.beta = kwargs.get('beta', 1000)

    def __call__(self, step):
        return self.beta

class PowerScheduler(object):
    def __init__(self, **kwargs):
        self.beta = kwargs.get('beta', 0.9)
        self.increasing = kwargs.get ('increasing', True)
        self.max_value = kwargs.get('max_value', 1)

        self.waiting_steps = kwargs.get('waiting_steps', 200)

    def __call__(self, step):
        if step < self.waiting_steps:
            step = 0
        else:
            step -= self.waiting_steps

        if self.increasing == False:
            return self.max_value * min(1., self.beta ** step)
        else:
            return self.max_value * min(1., 1.-self.beta ** step)


class PeriodicScheduler(object):
    """
    Periodically resets to 0.
    """

    def __init__(self, **kwargs):
        self.period_length = kwargs.get('period_length', 200)
        self.quarter_period_length = self.period_length * 0.25

        self.max_value = kwargs.get('max_value', 1.0)

        self.waiting_steps = kwargs.get('waiting_steps', 0)

    def __call__(self, step):
        step = max(0, step - self.waiting_steps)
        step = step % self.period_length
        if step < self.period_length * .5:
            return 0
        elif step < self.period_length * .75:
            return (step - 2 * self.quarter_period_length) / self.quarter_period_length * self.max_value
        else:
            return self.max_value


