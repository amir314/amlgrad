from typing import Dict, List, Generator

import numpy as np

from amlgrad.tensor import Tensor


class Layer():
    """
    This is the base class for layers of neural networks.

    A layer is simply a function. Neural networks are just concatinations
    of layers.
    """

    def __init__(self, params:Dict[str, Tensor]=None) -> None:
        self._params = {} if params is None else params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        self._params = new_params

    def forward(self, inputs:Tensor) -> Tensor:
        """
        Override this method to specify the output of a layer, given a batch of Tensors 'inputs'
        as an input to the model.
        """

        raise NotImplementedError


class NeuralNet():
    """
    This is the base class for neural networks.

    A Neural network consists of layers, see the Layer base class.
    """

    def __init__(self, layers:List[Layer]=None) -> None:
        self.layers = [] if layers is None else layers

    def parameters(self) -> Generator[Tensor]:
        for layer in self.layers:
            for param in layer.params.values():
                yield param

    def forward(self, inputs:Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


class Linear(Layer):
    def __init__(self, in_features:int, out_features:int) -> None:
        params = {}
        params['w'] = Tensor(
            data = np.random.randn(in_features, out_features)
        )
        params['b'] = Tensor(
            data = np.random.randn(out_features)
        )

        super().__init__(params)

    def forward(self, inputs:Tensor) -> Tensor:
        """
        Given a batch of Tensors 'inputs' of shape (batch_size, in_features),
        returns the Tensor input @ self.params['w'] + self.params['b'],
        which has shape (batch_size, out_features.)
        """

        return inputs @ self.params['w'] + self.params['b']


class ReLU(Layer):
    """
    This class implements the 'Rectified Linear Unit' activation function.
    """

    def __init__(self) -> None:
        super().__init__(self)

    def forward(self, inputs:Tensor) -> Tensor:
        """
        Applies the ReLU function to a Tensor element-wise.
        The ReLU function is defined as ReLU(x) = max(x, 0).
        """

        return inputs.relu()
