from typing import Callable

import numpy as np


class Tensor:
    def __init__(self, 
                 data,
                 requires_grad = False,
                 op:str=None,
                 children:set['Tensor']=set([])) -> None:
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.op = op
        self.children = children

        self.shape = self.data.shape

        if requires_grad:
            self.grad = Tensor(data=np.zeros(self.shape))
            self.grad_func:Callable[[None], None] =  lambda: None

    def __getitem__(self, idx):
        return self.data[idx]
    
    def __repr__(self):
        return f'Tensor(data={self.data}, requires_grad={self.requires_grad})'

    def __add__(self, other:'Tensor') -> 'Tensor':
        pass

    def backward(self, grad:'Tensor'=None) -> None:
        """
        This method is meant to be called on 0-dimensional Tensors. It will traverse the Tensor's 
        computation graph and calculate the gradient attributes of all Tensors in the graph along the way.

        If one wishes to call it on a non-0-dimensional Tensor, it should be made sure that the grad parameter 
        is not None and of the correct shape.
        """

        assert self.requires_grad, "Called backward on a Tensor with requires_grad set to False."

        if grad is None:
            assert self.shape == (), "backward should only be called on 0-dimensional Tensors."
            self.grad = Tensor(1) 
        else:
            self.grad = grad

        self.grad_func()

    def zero_grad(self) -> None:
        """
        Sets the grad attribute of every Tensor in the computation graph to zero. 
        """

        self.grad = 0
        for child in self.children:
            child.zero_grad()

    def sum(self) -> 'Tensor':
        """
        Sums up all the entries of this Tensor and returns a 0-dimensional Tensor.
        """

        return tensor_sum(self)
    

def tensor_sum(t: 'Tensor') -> 'Tensor':
    """
    Sums up all the entries in t and returns a 0-dimensional Tensor. 
    """
    
    out_data = t.data.sum()
    out_op = 'tensor_sum'
    out_children = set([t])
    
    out = Tensor(out_data,
                 t.requires_grad,
                 out_op,
                 out_children)
    
    def grad_func() -> None:
        """
        Let f(x) be a function and let x = out = tensor_sum(t). Then
            - df/dt = df/d(out)*d(out)/dt 
        The first factor is stored in out's grad attribute, the second factor is simply a tensor of ones with shape t.shape.
        """
        t.grad.data += out.grad.data 
        t.grad_func()
    out.grad_func = grad_func
    
    return out


if __name__=='__main__':
    t1 = Tensor([1,2,3], requires_grad=True)
    t2 = tensor_sum(t1)
    t2.backward()