from typing import Callable, Set

import numpy as np


class Tensor:
    """
    This class implements the Tensor, which essentially wraps data
    into a numpy nd-array and tracks computations between
    Tensors through the 'children' and 'op' attributes, through
    which a 'directed acyclic computation graph' (DAG) is built
    up internally. This allows differentiation of 0-dimensional 
    Tensors (e.g. as the output of a loss functions) with respect
    to all the Tensors before it in the DAG (e.g. parameters
    of a neural network), by calling the 'backward' method.
    These derivatives get stored in the respective Tensor's
    'grad' attribute.
    """

    def __init__(self,
                 data,
                 requires_grad:bool=False,
                 op:str=None,
                 children:Set['Tensor']=None) -> None:
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.op = op
        self.children = set([]) if children is None else children

        self.shape = self.data.shape
        self.dtype = self.data.dtype
        self.size = self.data.size

        self.grad = Tensor(data=np.zeros(self.data.shape)) if requires_grad else None
        self.grad_func:Callable[[None], None] = lambda: None

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other:'Tensor') -> 'Tensor':
        return tensor_add(self, other)

    def __sub__(self, other:'Tensor') -> 'Tensor':
        return tensor_sub(self, other)

    def __pow__(self, a:float) -> 'Tensor':
        return tensor_pow(self, a)

    def __mul__(self, other:'Tensor') -> 'Tensor':
        return tensor_mul(self, other)

    def __matmul__(self, other:'Tensor') -> 'Tensor':
        return tensor_matmul(self, other)

    def relu(self) -> 'Tensor':
        """
        Applies the ReLU (Rectified Linear Unit) function to a Tensor element-wise
        and returns the resulting Tensor.

        The ReLU function is defined as ReLU(x) = max(x, 0).
        """

        return tensor_relu(self)

    def backward(self, grad:'Tensor'=None) -> None:
        """
        This method is meant to be called on 0-dimensional Tensors. It will traverse the Tensor's
        computation graph and calculate the gradient attributes of all Tensors in the graph along
        the way.

        If one wishes to call this method on a non-0-dimensional Tensor,
        it should be made sure that the 'grad' parameter is not None and of the correct shape.
        """

        assert self.requires_grad, "Called backward on a Tensor with requires_grad set to False."

        if grad is None:
            assert self.shape == (), "Called backward with grad=None on non-0-dim Tensor."
            self.grad = Tensor(1)
        else:
            assert grad.shape == self.shape, "grad.shape must match self.shape."
            self.grad = grad

        self.grad_func()

    def zero_grad(self) -> None:
        """
        Sets the gradient of every Tensor in the computation graph to zero.
        """

        if self.grad:
            self.grad.data = np.zeros_like(self.grad.data)

            for child in self.children:
                child.zero_grad()

    def sum(self) -> 'Tensor':
        """
        Sums up all the entries of a Tensor and returns the resulting
        0-dimensional Tensor.
        """

        return tensor_sum(self)

    def mean(self) -> 'Tensor':
        """
        Returns the mean of all values in the Tensor and returns
        the resulting 0-dimensional Tensor.
        """

        return Tensor(1/self.size)*self.sum()


def tensor_sum(t:'Tensor') -> 'Tensor':
    """
    Sums up all the entries in a Tensor and returns the resulting 
    0-dimensional Tensor.
    """

    out_data = t.data.sum()
    out_requires_grad = t.requires_grad
    out_op = 'tensor_sum'
    out_children = set([t])

    out = Tensor(out_data,
                 out_requires_grad,
                 out_op,
                 out_children)

    def grad_func() -> None:
        """
        Let f(x) be a scalar-valued function and let x = out = tensor_sum(t). 
        Then, by the chain rule:
            - df/dt = df/d(out) * d(out)/dt.

        The first factor is stored in out's grad attribute, the second factor is 
        simply a tensor of ones with shape t.shape.
        """

        if t.requires_grad:
            t.grad.data += out.grad.data * np.ones_like(t.grad.data)
            t.grad_func()

    out.grad_func = grad_func

    return out

def tensor_add(t1:'Tensor', t2:'Tensor') -> 'Tensor':
    """
    Adds up two Tensors element-wise and returns the resulting Tensor.
    """

    out_data = t1.data + t2.data
    out_requires_grad = t1.requires_grad or t2.requires_grad
    out_op = '+'
    out_children = set([t1, t2])

    out = Tensor(out_data,
                 out_requires_grad,
                 out_op,
                 out_children)

    def grad_func() -> None:
        """
        Let f(x) be a scalar-valued function and let x = out = tensor_add(t1, t2).
        Then, by the chain rule:
            - df/dt1 = df/d(out) * d(out)/dt1.

        The first factor is stored in out's grad attribute, the second gets calculated 
        in t1.grad_func.
        The way the second factor is handled depends on whether or not
        numpy's broadcasting is employed, when adding up the data
        of t1 and t2:
            - If t1.shape == t2.shape, then t1.grad.data is simply
            np.ones_like(t1).
            - If, say, t1.shape = (1, something) and t2.shape = (n, something),
            then broadcasting is employed. Thus, a change of dx in one entry of
            t1 causes a change of dx in n entries of out = tensor_add(t1, t2).
            As a consequence, after the resulting Tensor gets fed into f, a change of dx
            in an entry of t1 results in a change of n*dx in f.
            We deal with this by summing across all dimensions of out,
            in which broadcasting was employed.

        The t2 case is analogous.
        """

        if t1.requires_grad:
            grad = out.grad.data * np.ones_like(out.grad.data)
            excess_dims = len(grad.shape) - len(t1.shape) # len(t1.shape) <= len(out.shape)

            # Sum out excess dims
            if excess_dims > 0:
                for _ in range(excess_dims):
                    grad = np.sum(grad, axis=0)

            # Sum over all dimensions that were broadcasted
            for dim, n in enumerate(t1.shape):
                if grad.shape[dim] - n > 0:
                    grad = np.sum(grad, axis=dim, keepdims=True)

            t1.grad.data += grad

        if t2.requires_grad:
            grad = out.grad.data * np.ones_like(out.grad.data)
            excess_dims = len(grad.shape) - len(t2.shape) # len(t2.shape) <= len(out.shape)

            # Sum out excess dims
            if excess_dims > 0:
                for _ in range(excess_dims):
                    grad = np.sum(grad, axis=0)

            # Sum across all dimensions that were broadcasted
            for dim, n in enumerate(t2.shape):
                if grad.shape[dim] - n > 0:
                    grad = np.sum(grad, axis=dim, keepdims=True)

            t2.grad.data += grad

        for child in out.children:
            if child.requires_grad:
                child.grad_func()

    out.grad_func = grad_func

    return out

def tensor_sub(t1:'Tensor', t2:'Tensor') -> 'Tensor':
    """
    Subracts two Tensors from each other element-wise
    and returns the resulting Tensor.
    """

    out_data = t1.data - t2.data
    out_requires_grad = t1.requires_grad or t2.requires_grad
    out_op = '-'
    out_children = set([t1, t2])

    out = Tensor(out_data,
                 out_requires_grad,
                 out_op,
                 out_children)

    def grad_func() -> None:
        """
        The logic here is the same to how it is for tensor_add, except, of course,
        a slight difference in the derivative d(out)/dt2 because of the minus sign.
        """

        if t1.requires_grad:
            grad = out.grad.data * np.ones_like(out.grad.data)
            excess_dims = len(grad.shape) - len(t1.shape) # len(t1.shape) <= len(out.shape)

            # Sum out excess dims
            if excess_dims > 0:
                for _ in range(excess_dims):
                    grad = np.sum(grad, axis=0)

            # Sum over all dimensions that were broadcasted
            for dim, n in enumerate(t1.shape):
                if grad.shape[dim] - n > 0:
                    grad = np.sum(grad, axis=dim, keepdims=True)

            t1.grad.data += grad

        if t2.requires_grad:
            grad = out.grad.data * -1 * np.ones_like(out.grad.data)
            excess_dims = len(grad.shape) - len(t2.shape) # len(t2.shape) <= len(out.shape)

            # Sum out excess dims
            if excess_dims > 0:
                for _ in range(excess_dims):
                    grad = np.sum(grad, axis=0)

            # Sum across all dimensions that were broadcasted
            for dim, n in enumerate(t2.shape):
                if grad.shape[dim] - n > 0:
                    grad = np.sum(grad, axis=dim, keepdims=True)

            t2.grad.data += grad

        for child in out.children:
            if child.requires_grad:
                child.grad_func()

    out.grad_func = grad_func

    return out

def tensor_mul(t1:'Tensor', t2:'Tensor') -> 'Tensor':
    """
    Multiplies two Tensors with each other element-wise
    and returns the resulting Tensor.
    """

    out_data = t1.data * t2.data
    out_requires_grad = t1.requires_grad or t2.requires_grad
    out_op = '*'
    out_children = set([t1, t2])

    out = Tensor(out_data,
                 out_requires_grad,
                 out_op,
                 out_children)

    def grad_func() -> None:
        """
        Let f(x) be a scalar-valued function and let x = out = tensor_mul(t1, t2).
        Then, by the chain rule:
            - df/dt1 = df/d(out) * d(out)/dt1.

        The first factor is stored in out's grad attribute, the second gets calculated 
        in t1.grad_func.
        The way the second factor is handled depends on whether or not
        numpy's broadcasting is employed, when multiplying up the data
        of t1 and t2:
            - If t1.shape == t2.shape, then t1.grad.data is simply
            t2.data.
            - If, say, t1.shape = (1, something) and t2.shape = (n, something),
            then broadcasting is employed. Let's say one of the entries of 
            out = tensor_mul(t1, t2) is given by a1*a2, where a1 comes from t1 and 
            a2 from t2. Then, a change of dx in a1 causes a change of dx*a2 in n 
            entries of out. As a consequence, after out gets fed into f, a 
            change of dx in a1 results in a change of n*dx*a2 in f.
            We deal with this by summing across all dimensions of out,
            in which broadcasting was employed.

        The t2 case is analogous.
        """

        if t1.requires_grad:
            grad = out.grad.data * t2.data
            excess_dims = len(grad.shape) - len(t1.shape) # len(t1.shape) <= len(out.shape)

            # Sum out excess dims
            if excess_dims > 0:
                for _ in range(excess_dims):
                    grad = np.sum(grad, axis=0)

            # Sum over all dimensions that were broadcasted
            for dim, n in enumerate(t1.shape):
                if grad.shape[dim] - n > 0:
                    grad = np.sum(grad, axis=dim, keepdims=True)

            t1.grad.data += grad

        if t2.requires_grad:
            grad = out.grad.data * t1.data
            excess_dims = len(grad.shape) - len(t2.shape) # len(t2.shape) <= len(out.shape)

            # Sum out excess dims
            if excess_dims > 0:
                for _ in range(excess_dims):
                    grad = np.sum(grad, axis=0)

            # Sum over all dimensions that were broadcasted
            for dim, n in enumerate(t2.shape):
                if grad.shape[dim] - n > 0:
                    grad = np.sum(grad, axis=dim, keepdims=True)

            t2.grad.data += grad

        for child in out.children:
            if child.requires_grad:
                child.grad_func()

    out.grad_func = grad_func

    return out

def tensor_matmul(t1:'Tensor', t2:'Tensor') -> 'Tensor':
    """
    Calculates the matrix-product of two Tensors and returns the resulting Tensor.
    Expects Tensors of dimension greater than or equal to 1 and such that the
    last dimension of t1 is the same as the second-to-last of t2.
    """

    out_data = t1.data @ t2.data
    out_requires_grad = t1.requires_grad or t2.requires_grad
    out_op = '@'
    out_children = set([t1, t2])

    out = Tensor(out_data,
                 out_requires_grad,
                 out_op,
                 out_children)

    def grad_func() -> None:
        """
        Let f(x) be a scalar-valued function and let x = out = tensor_matmul(t1, t2).
        Then, by the chain rule:
            - df/dt1 = df/d(out) @ transpose(d(out)/dt1).

        The first factor is stored in out's grad attribute, the second gets calculated 
        in t1.grad_func.
        The way the second factor is handled depends on whether or not
        numpy's broadcasting is employed, when calculating the matrix-product
        of t1.data and t2.data.

        Let's first focus on 1d- and 2d-Tensors (i.e. matrices).
            - If t1.shape == (n, k) and t2.shape == (k, m), then out.shape = (n, m)
            and d(out)/dt1 = np.transpose(t2.data).
            - If t1.shape == (k,) and t2.shape == (k, m), then out.shape = (1, m) and,
            again, d(out)/dt1 = np.transpose(t2.data).

        If (at least) one of the Tensors has dimension > 2, then broadcasting is employed
        and we need to sum out/across the broadcasted dimensions, similarly to how we
        do for the other tensor operations like tensor_mul and tensor_add.

        The t2 case is similar. TODO: Explain t2 case in some detail.
        """

        if t1.requires_grad:
            grad = out.grad.data @ np.swapaxes(t2.data, -1, -2) # grad.shape[-2:] = t1.shape[-2:]
            excess_dims = len(grad.shape) - len(t1.shape)

            # Sum out excess dims
            if excess_dims > 0:
                for _ in range(excess_dims):
                    grad = np.sum(grad, axis=0)

            # Sum over all dimensions that were broadcasted
            for dim, n in enumerate(t1.shape):
                if grad.shape[dim] - n > 0:
                    grad = np.sum(grad, axis=dim, keepdims=True)

            t1.grad.data += grad

        if t2.requires_grad:
            grad = np.swapaxes(t1.data, -1, -2) @ out.grad.data # grad.shape[-2:] = t2.shape[-2:]
            excess_dims = len(grad.shape) - len(t2.shape)

            # Sum out excess dims
            if excess_dims > 0:
                for _ in range(excess_dims):
                    grad = np.sum(grad, axis=0)

            # Sum over all dimensions that were broadcasted
            for dim, n in enumerate(t2.shape):
                if grad.shape[dim] - n > 0:
                    grad = np.sum(grad, axis=dim, keepdims=True)

            t2.grad.data += grad

        for child in out.children:
            if child.requires_grad:
                child.grad_func()

    out.grad_func = grad_func

    return out

def tensor_pow(t:'Tensor', a:float) -> 'Tensor':
    """
    Calculates the element-wise power of a Tensor to a float
    and returns the resulting Tensor.
    """

    out_data = t.data ** a
    out_requires_grad = t.requires_grad
    out_op = '**'
    out_children = set([t])

    out = Tensor(out_data,
                 out_requires_grad,
                 out_op,
                 out_children)

    def grad_func() -> None:
        """
        Let f(x) be a scalar-valued function and let x = out = tensor_pow(t, a).
        Then, by the chain rule:
            - df/dt = df/d(out)*d(out)/dt.

        The first factor is stored in out's grad attribute, the second gets calculated 
        in t.grad_func. If g(x) = x ** a, then g'(x) = a * (x ** (a-1)) and hence
        d(out)/dt = a * ( t.data ** (a-1) ).
        """

        if t.requires_grad:
            t.grad.data += out.grad.data * ( a * ( t.data ** (a-1) ) ) if not a == 0 else 0
            t.grad_func()

    out.grad_func = grad_func

    return out

def tensor_relu(t:'Tensor') -> 'Tensor':
    """
    Applies the ReLU (Rectified Linear Unit) function to a Tensor element-wise
    and returns the resulting Tensor.

    The ReLU function is defined as ReLU(x) = max(x, 0).
    """

    out_data = t.data * (t.data > 0)
    out_requires_grad = True
    out_op = 'ReLU'
    out_children = set([t])

    out = Tensor(out_data,
                 out_requires_grad,
                 out_op,
                 out_children)

    def grad_func() -> None:
        """
        Let f(x) be a scalar-valued function and let x = out = relu(t). 
        Then, by the chain rule:
            - df/dt = df/d(out) * d(out)/dt.

        The first factor is stored in out's grad attribute, the second factor is 
        simply a tensor of ones wherever t is positive and 0's else.
        """

        if t.requires_grad:
            t.grad.data += out.grad.data * (t.data > 0)
            t.grad_func()

    out.grad_func = grad_func

    return out
