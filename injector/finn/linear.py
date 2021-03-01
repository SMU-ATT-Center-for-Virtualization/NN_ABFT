import torch.nn as nn


class FILinear(nn.Linear):
    """
    A fault injector wrapper module for nn.Linear transformation.

    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """

    def __init__(self, fi, name, in_features, out_features, weight=None, bias=None):
        self.fi = fi
        self.name = name
        self.id = fi.register_layer(name, FILinear)

        super(FILinear, self).__init__(in_features, out_features, bias is not None)

        if weight is not None:
            self.weight = weight
        if bias is not None:
            self.bias = bias

    def forward(self, input):
        if self.fi.enable_injection and self.id == self.fi.to_layer:

            if self.fi.enable_inject_features:
                fault_data = self.fi.inject_features(input.data)
                for idx, (indices, fault_val) in enumerate(fault_data):
                    input.data[tuple([idx] + indices)] = fault_val

            weight_copy = self.weight.clone()

            if self.fi.enabel_inject_weights:
                indices, fault_val = self.fi.inject(weight_copy.data)
                weight_copy.data[tuple(indices)] = fault_val

            return nn.functional.linear(input, weight_copy, self.bias)
        else:
            return super(FILinear, self).forward(input)

    @staticmethod
    def from_pytorch_impl(fi, name, linear: nn.Linear):
        return FILinear(
            fi,
            name,
            linear.in_features,
            linear.out_features,
            linear.weight,
            linear.bias,
        )

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, id={self.id})'
