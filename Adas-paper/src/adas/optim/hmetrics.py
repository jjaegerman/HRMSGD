"""
"""
from typing import List, Union, Tuple

import sys

import numpy as np
import torch

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .components import LayerMetrics, ConvLayerMetrics
    from .matrix_factorization import EVBMF
else:
    from optim.components import LayerMetrics, ConvLayerMetrics
    from optim.matrix_factorization import EVBMF


class Metrics:
    def __init__(self, params) -> None:
        '''
        parameters: list of torch.nn.Module.parameters()
        '''
        self.params = params
        self.history = list()
        mask = list()
        #create mask
        self.mask = set(mask)

    def __call__(self) -> List[Tuple[int, Union[LayerMetrics,
                                                ConvLayerMetrics]]]:
        '''
        Computes the knowledge gain (S) and mapping condition (condition)
        '''
        metrics: List[Tuple[int, Union[LayerMetrics,
                                       ConvLayerMetrics]]] = list()
        for layer_index, layer in enumerate(self.params):
            if layer_index in self.mask:
                metrics.append((layer_index, None))
                continue
            # if np.less(np.prod(layer.shape), 10_000):
            #     metrics.append((layer_index, None))
            if len(layer.shape) == 4:
                layer_tensor = layer.data
                tensor_size = layer_tensor.shape
                mode_3_unfold = layer_tensor.permute(1, 0, 2, 3)
                mode_3_unfold = torch.reshape(
                    mode_3_unfold, [tensor_size[1], tensor_size[0] *
                                    tensor_size[2] * tensor_size[3]])
                mode_4_unfold = layer_tensor
                mode_4_unfold = torch.reshape(
                    mode_4_unfold, [tensor_size[0], tensor_size[1] *
                                    tensor_size[2] * tensor_size[3]])
                in_rank, in_KG, in_condition = self.compute_low_rank(
                    mode_3_unfold, tensor_size[1])
                if in_rank is None and in_KG is None and in_condition is None:
                    if len(self.history) > 0:
                        in_rank = self.history[-1][
                            layer_index][1].input_channel.rank
                        in_KG = self.history[-1][
                            layer_index][1].input_channel.KG
                        in_condition = self.history[-1][
                            layer_index][1].input_channel.condition
                    else:
                        in_rank = in_KG = in_condition = 0.
                out_rank, out_KG, out_condition = self.compute_low_rank(
                    mode_4_unfold, tensor_size[0])
                if out_rank is None and out_KG is None and out_condition is None:
                    if len(self.history) > 0:
                        out_rank = self.history[-1][
                            layer_index][1].output_channel.rank
                        out_KG = self.history[-1][
                            layer_index][1].output_channel.KG
                        out_condition = self.history[-1][
                            layer_index][1].output_channel.condition
                    else:
                        out_rank = out_KG = out_condition = 0.
                metrics.append((layer_index, ConvLayerMetrics(
                    input_channel=LayerMetrics(
                        rank=in_rank,
                        KG=in_KG,
                        condition=in_condition),
                    output_channel=LayerMetrics(
                        rank=out_rank,
                        KG=out_KG,
                        condition=out_condition))))
            elif len(layer.shape) == 2:
                rank, KG, condition = self.compute_low_rank(
                    layer, layer.shape[0])
                if rank is None and KG is None and condition is None:
                    if len(self.history) > 0:
                        rank = self.history[-1][layer_index][1].rank
                        KG = self.history[-1][layer_index][1].KG
                        condition = self.history[-1][layer_index][1].condition
                    else:
                        rank = KG = condition = 0.
                metrics.append((layer_index, LayerMetrics(
                    rank=rank,
                    KG=KG,
                    condition=condition)))
            else:
                metrics.append((layer_index, None))
        self.history.append(metrics)
        return metrics
