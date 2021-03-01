import torch.nn as nn

from injector.finn.matmul import Matmul, BatchMatmul, FIMatmul, FIBatchMatmul
from injector.finn.linear import FILinear

__all__ = ('Matmul', 'BatchMatmul', 'FIMatmul', 'FIBatchMatmul', 'FILinear', 'Print')


class Print(nn.Module):
    """Utility module to temporarily replace modules to assess activation shape.
    This is useful, e.g., when creating a new model and you want to know the size
    of the input to nn.Linear and you want to avoid calculating the shape by hand.

    From https://github.com/IntelLabs/distiller/blob/master/distiller/modules/__init__.py
    """

    def forward(self, x):
        print(x.size())
        return x
