import torch
import torch.nn as nn


class Matmul(nn.Module):
    """
    A wrapper module for matmul operation between 2 tensors.

    From https://github.com/IntelLabs/distiller/blob/master/distiller/modules/matmul.py
    """

    def __init__(self, *args, **kwargs):
        super(Matmul, self).__init__(*args, **kwargs)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return a.matmul(b)


class BatchMatmul(nn.Module):
    """
    A wrapper module for torch.bmm operation between 2 3-D tensors.

    From https://github.com/IntelLabs/distiller/blob/master/distiller/modules/matmul.py
    """

    def __init__(self, *args, **kwargs):
        super(BatchMatmul, self).__init__(*args, **kwargs)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        return torch.bmm(a, b)


class FIMatmul(Matmul):
    """
    A fault injector wrapper module for torch.matmul operation between 2 tensors.

    https://pytorch.org/docs/stable/generated/torch.matmul.html
    """

    def __init__(self, fi, name, *args, **kwargs):
        super(FIMatmul, self).__init__(*args, **kwargs)
        self.fi = fi
        self.name = name
        self.id = fi.register_layer(name, FIMatmul)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if self.fi.enable_injection and self.id == self.fi.to_layer:
            a_copy = a.clone()
            indices, fault_val = self.fi.inject(a_copy.data)
            a_copy.data[tuple(indices)] = fault_val
            return a_copy.matmul(b)
        return super(FIMatmul, self).forward(a, b)

    @staticmethod
    def from_pytorch_impl(fi, name, matmul: Matmul):
        return FIMatmul(fi, name)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, id={self.id})'


class FIBatchMatmul(BatchMatmul):
    """
    A fault injector wrapper module for torch.bmm operation between 2 3-D tensors.

    https://pytorch.org/docs/stable/generated/torch.bmm.html
    """

    def __init__(self, fi, name, *args, **kwargs):
        super(FIBatchMatmul, self).__init__(*args, **kwargs)
        self.fi = fi
        self.name = name
        self.id = fi.register_layer(name, FIBatchMatmul)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if self.fi.enable_injection and self.id == self.fi.to_layer:
            a_copy = a.clone()
            indices, fault_val = self.fi.inject(a_copy.data)
            a_copy.data[tuple(indices)] = fault_val
            return torch.bmm(a_copy, b)
        return super(FIBatchMatmul, self).forward(a, b)

    @staticmethod
    def from_pytorch_impl(fi, name, batchmatmul: BatchMatmul):
        return FIBatchMatmul(fi, name)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, id={self.id})'
