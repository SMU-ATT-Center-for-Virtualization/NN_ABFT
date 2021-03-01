from bitstring import BitArray
import numpy as np
import torch


class TensorInjector:
    @classmethod
    def get_bitarray(cls, data: torch.Tensor):
        length = data.element_size() * 8
        if data.is_floating_point():
            return BitArray(float=data, length=length)
        else:
            return BitArray(int=data, length=length)

    @classmethod
    def bit_flip(cls, data: torch.Tensor, loc: int = None):
        fault_data = cls.get_bitarray(data)
        if loc is None:
            loc = np.random.randint(0, fault_data.len)
        fault_data.invert(loc)
        return fault_data.float if data.is_floating_point() else fault_data.int

    @classmethod
    def stuck_at(cls, data: torch.Tensor, loc: int = None, stuck_at: int = 0):
        fault_data = cls.get_bitarray(data)
        stuck_bit = '0b0' if stuck_at == 0 else '0b1'
        if loc is None:
            loc = np.random.randint(0, fault_data.len)
        fault_data.overwrite(stuck_bit, loc)
        return fault_data.float if data.is_floating_point() else fault_data.int

    @classmethod
    def random_select(cls, data: torch.Tensor):
        indices = tuple(map(np.random.randint, data.shape))
        return indices, data[indices]

    @classmethod
    def random_select_nonzero(cls, data: torch.Tensor):
        tried = 0
        while tried < data.numel():
            indices, data_val = cls.random_select(data)
            if data_val != 0.0:
                break
        return indices, data_val
