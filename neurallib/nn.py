from typing import List, Generator

import numpy as np

from amlgrad.neurallib.tensor import Tensor


class Layer():
    """
    This is the base class for layers of neural networks.

    A layer is simply a function. Neural networks are just concatinations
    of layers.
    """

    def __init__(self, params:List[Tensor]=None) -> None:
        self.params = [] if params is None else params

        for param in self.params:
            assert param.requires_grad, "Layer contains parameter with requires_grad set to False."

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

    def parameters(self) -> Generator[Tensor, None, None]:
        """
        Returns a generator object, which contains all parameters
        of the neural network.
        """

        for layer in self.layers:
            for param in layer.params:
                yield param

    def forward(self, inputs:Tensor) -> Tensor:
        """
        Propagates the input through the layers in the neural network
        and returns the resulting Tensor.
        """

        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs


class Linear(Layer):
    def __init__(self, in_features:int, out_features:int) -> None:
        self.w = Tensor(
            data = np.random.randn(in_features, out_features),
            requires_grad=True
        )
        self.b = Tensor(
            data = np.random.randn(out_features),
            requires_grad=True
        )
        params = [self.w, self.b]

        super().__init__(params)

    def forward(self, inputs:Tensor) -> Tensor:
        """
        Given a batch of Tensors 'inputs' of shape (batch_size, in_features),
        returns the Tensor (batch_size, out_features) by applying an affine
        linear transformation to each feature in the batch.
        """

        return inputs @ self.w + self.b


class Conv1d(Layer):
    pass


class ReLU(Layer):
    """
    This class implements the 'Rectified Linear Unit' activation function.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs:Tensor) -> Tensor:
        """
        Applies the ReLU function to a Tensor element-wise.
        The ReLU function is defined as ReLU(x) = max(x, 0).
        """

        return inputs.relu()
