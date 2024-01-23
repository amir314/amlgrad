from typing import Iterable

from amlgrad.neurallib.tensor import Tensor


class Optimizer():
    """
    This is the base class for optimizers.

    An optimizer is an algorithm, which is used to update
    the parameters of a model, in order to minimize
    a loss function.
    """

    def __init__(self,
                 params:Iterable[Tensor],
                 lr:float=0.001) -> None:
        """
        An optimizer needs a model's parameters and a learning rate (lr), which
        controls the size of the parameter update rule. Standard lr values are 0.01
        or 0.001.
        """

        self.params = params
        self.lr = lr

    def step(self) -> None:
        """
        Override this method to implement the update rule of an optimizer.
        """

        raise NotImplementedError


class SGD(Optimizer):
    """
    This class implements the 'Stochastic Gradient Descent' optimizer.
    """

    def __init__(self,
                 params:Iterable[Tensor],
                 lr:float) -> None:
        super().__init__(params, lr)

    def step(self) -> None:
        """
        The SGD update rule is given by:

            new_param = param - lr * param.grad.
        """

        for param in self.params:
            param -= self.lr*param.grad
