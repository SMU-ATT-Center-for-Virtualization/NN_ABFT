import torch
import numpy as np

from injector import TensorInjector
from injector.finn import FILinear


class ModelInjector:
    ERROR_TYPES = (
        'random_bit_flip_percentage',
        'random_bit_flip_number',
        'stuck_at_0',
        'stuck_at_1',
        'bit_flip_at_location',
        'missing_node',
        'missing_connection',
        'zero_weight',
    )

    def __init__(
        self,
        model: torch.nn.Module,
        enable_injection: bool = True,
        to_loc: int = None,
        to_epoch: int = None,
        to_layer: int = 0,
        enable_inject_features: bool = True,
        enabel_inject_weights: bool = True,
        error_rate: float = 1.0,
        error_number: int = 1,
        error_type: str = None,
        error_args: dict = None,
    ):
        self.model = model
        self.enable_injection = enable_injection
        self.to_loc = to_loc
        self.to_epoch = to_epoch
        self.to_layer = to_layer
        self.enable_inject_features = enable_inject_features
        self.enabel_inject_weights = enabel_inject_weights
        self.error_rate = error_rate
        self.error_number = error_number
        assert (
            error_type is None or error_type in self.ERROR_TYPES
        ), f'Only support error_type as {self.ERROR_TYPES}'
        self.error_type = error_type
        self.error_args = error_args

        self.injection_count = 0
        self.layer_supported = {}
        self.last_layer_index = -1
        self.layer_refs = []

    def patch_model(self):
        for name, layer in self.model.named_children():
            patched_layer = self.patch_layer(layer, name)
            setattr(self.model, name, patched_layer)
            self.patch_model(layer)  # recursively patch block layers

    def patch_layer(self, layer, layer_name):
        if self.enable_injection:
            layer_type = type(layer)
            if layer_type in self.layer_supported:
                return self.layer_supported[layer_type].from_pytorch_impl(
                    self, layer_name, layer
                )
        return layer

    def register_layer(self, layer_name, layer_type):
        self.last_layer_index += 1
        self.layer_refs.append((layer_name, layer_type))
        return self.last_layer_index

    def inject_features(self, data, batch_size):
        fault_data = []
        for batch_idx in range(batch_size):
            amount = self.error_number
            if 'percentage' in self.error_type:
                amount = int(data[batch_idx].numel() * self.error_rate)
            for _ in range(amount):
                indices, fault_val = self.inject_tensor(data)
                fault_data.append(([batch_idx] + indices, fault_val))
        return fault_data

    def inject_weights(self, data):
        return self.inject(data)

    def inject_tensor(self, data):
        indices, data_val = TensorInjector.random_select_nonzero(data)
        fault_val = self.inject_val(data_val)
        self.injection_count += 1
        return indices, fault_val

    def inject_val(self, data):
        if 'bit_flip' in self.error_type:
            if self.error_type == 'bit_flip_at_location':
                return TensorInjector.bit_flip(data, to_loc)
            return TensorInjector.bit_flip(data)
        elif 'stuck_at' in self.error_type:
            return TensorInjector.stuck_at(
                data, stuck_at=0 if self.error_type == 'stuck_at_0' else 1
            )
