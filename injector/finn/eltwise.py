import torch
import torch.nn as nn


class EltwiseAdd(nn.Module):
    """
    A wrapper module for element-wise addition.
    """

    def __init__(self, *args, inplace=False, **kwargs):
        super(EltwiseAdd, self).__init__(*args, **kwargs)
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = res + t
        return res


class EltwiseSub(nn.Module):
    """
    A wrapper module for element-wise subtraction.
    """

    def __init__(self, *args, inplace=False, **kwargs):
        super(EltwiseSub, self).__init__(*args, **kwargs)
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res -= t
        else:
            for t in input[1:]:
                res = res - t
        return res


class EltwiseMult(nn.Module):
    """
    A wrapper module for element-wise multiplication.
    """

    def __init__(self, *args, inplace=False, **kwargs):
        super(EltwiseMult, self).__init__(*args, **kwargs)
        self.inplace = inplace

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = res * t
        return res


class EltwiseDiv(nn.Module):
    """
    A wrapper module for element-wise division.
    """

    def __init__(self, *args, inplace=False, **kwargs):
        super(EltwiseDiv, self).__init__(*args, **kwargs)
        self.inplace = inplace

    def forward(self, x: torch.Tensor, y):
        if self.inplace:
            return x.div_(y)
        return x.div(y)


class FIEltwiseAdd(EltwiseAdd):
    """
    A fault injector wrapper module for element-wise addition.
    """

    def __init__(self, fi, name, *args, inplace=False, **kwargs):
        super(FIEltwiseAdd, self).__init__(*args, inplace, **kwargs)
        self.fi = fi
        self.name = name
        self.id = fi.register_layer(name, FIEltwiseAdd)

    def forward(self, *input):
        if self.fi.enable_injection and self.id == self.fi.to_layer:
            res_copy = input[0].clone()
            indices, fault_val = self.fi.inject(res_copy.data)
            res_copy.data[tuple(indices)] = fault_val
            if self.inplace:
                for t in input[1:]:
                    res_copy += t
            else:
                for t in input[1:]:
                    res_copy = res_copy + t
            return res_copy
        else:
            return super(FIEltwiseAdd, self).forward(*input)

    @staticmethod
    def from_pytorch_impl(fi, name, eltwiseadd: EltwiseAdd):
        return FIEltwiseAdd(fi, name, eltwiseadd.inplace)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, inplace={self.inplace}, id={self.id})'


class FIEltwiseSub(EltwiseSub):
    """
    A fault injector wrapper module for element-wise addition.
    """

    def __init__(self, fi, name, *args, inplace=False, **kwargs):
        super(FIEltwiseSub, self).__init__(*args, inplace, **kwargs)
        self.fi = fi
        self.name = name
        self.id = fi.register_layer(name, FIEltwiseSub)

    def forward(self, *input):
        if self.fi.enable_injection and self.id == self.fi.to_layer:
            res_copy = input[0].clone()
            indices, fault_val = self.fi.inject(res_copy.data)
            res_copy.data[tuple(indices)] = fault_val
            if self.inplace:
                for t in input[1:]:
                    res_copy -= t
            else:
                for t in input[1:]:
                    res_copy = res_copy - t
            return res_copy
        else:
            return super(FIEltwiseSub, self).forward(*input)

    @staticmethod
    def from_pytorch_impl(fi, name, eltwisesub: EltwiseSub):
        return FIEltwiseSub(fi, name, eltwisesub.inplace)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, inplace={self.inplace}, id={self.id})'


class FIEltwiseMult(EltwiseMult):
    """
    A fault injector wrapper module for element-wise multiplication.
    """

    def __init__(self, fi, name, *args, inplace=False, **kwargs):
        super(FIEltwiseMult, self).__init__(*args, inplace, **kwargs)
        self.fi = fi
        self.name = name
        self.id = fi.register_layer(name, FIEltwiseMult)

    def forward(self, *input):
        if self.fi.enable_injection and self.id == self.fi.to_layer:
            res_copy = input[0].clone()
            indices, fault_val = self.fi.inject(res_copy.data)
            res_copy.data[tuple(indices)] = fault_val
            if self.inplace:
                for t in input[1:]:
                    res_copy *= t
            else:
                for t in input[1:]:
                    res_copy = res_copy * t
            return res_copy
        else:
            return super(FIEltwiseMult, self).forward(*input)

    @staticmethod
    def from_pytorch_impl(fi, name, eltwisemult: EltwiseMult):
        return FIEltwiseMult(fi, name, eltwisemult.inplace)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, inplace={self.inplace}, id={self.id})'


class FIEltwiseDiv(EltwiseDiv):
    """
    A fault injector wrapper module for element-wise division.
    """

    def __init__(self, fi, name, *args, inplace=False, **kwargs):
        super(FIEltwiseDiv, self).__init__(*args, inplace, **kwargs)
        self.fi = fi
        self.name = name
        self.id = fi.register_layer(name, FIEltwiseDiv)

    def forward(self, x: torch.Tensor, y):
        if self.fi.enable_injection and self.id == self.fi.to_layer:
            x_copy = x.clone()
            indices, fault_val = self.fi.inject(x_copy.data)
            x_copy.data[tuple(indices)] = fault_val
            if self.inplace:
                return fault_val.div_(y)
            return fault_val.div(y)
        else:
            return super(FIEltwiseDiv, self).forward(x, y)

    @staticmethod
    def from_pytorch_impl(fi, name, eltwisediv: EltwiseDiv):
        return FIEltwiseDiv(fi, name, eltwisediv.inplace)

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, inplace={self.inplace}, id={self.id})'
