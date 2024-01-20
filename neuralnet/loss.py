from amlgrad.tensor import Tensor


class Loss():
    """
    This is the base class for loss functions.

    A loss function quantifies the performance of a model. A loss close
    to zero is desirable. By minimizing the loss function with respect
    to a models parameters, the model learns to make better predictions.
    """

    def __init__(self) -> None:
        pass

    def forward(self, preds:Tensor, labels:Tensor) -> Tensor:
        """
        Override this method to specifiy the output of a loss function, given
        predicted values 'preds' and the corresponding actual values 'labels'.

        Must return a 0-dimensional Tensor.
        """

        raise NotImplementedError


class MSE(Loss):
    """
    This class implements the 'Mean Squared Error' loss function.
    """

    def __init__(self) -> None:
        pass

    def forward(self, preds:Tensor, labels:Tensor) -> Tensor:
        """
        Given predictions 'preds' and labels 'labels', the MSE is given by

        MSE(preds, labels) = mean( (preds - labels)**2 ).
        """

        return ( (preds - labels)**2 ).mean()
