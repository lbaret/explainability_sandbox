import random
from copy import deepcopy
from typing import Dict

import scipy.stats as stats
import torch
import torch.nn as nn


class NeuronsImportance:
    @staticmethod
    def neuron_is_important(samples: torch.Tensor, positive_indices: torch.Tensor, negative_indices: torch.Tensor) -> bool:
        is_important = False
        X_positive = samples[positive_indices]
        X_negative = samples[negative_indices]

        is_X1_normal = stats.shapiro(X_positive).pvalue > 0.05
        is_X2_normal = stats.shapiro(X_negative).pvalue > 0.05

        if is_X1_normal and is_X2_normal:
            is_important = stats.ttest_ind(X_positive, X_negative, equal_var=False).pvalue < 0.05
        else:
            is_important = stats.mannwhitneyu(X_positive, X_negative).pvalue < 0.05
        
        return is_important

    @staticmethod
    def mask_important_neurons(model: nn.Module, important_neurons: Dict[str, torch.Tensor], percentage_masked: float=0.1) -> nn.Module:
        masked_model = deepcopy(model)
        masked_model.eval()
        for layer_name, positive_indices in important_neurons.items():
            mask_size = int(positive_indices.shape[0] * percentage_masked)
            indices = torch.LongTensor(random.sample(positive_indices.tolist(), k=mask_size))

            layer = masked_model.get_submodule(layer_name)
            layer_weights = layer.weight.data
            layer_biases = layer.bias.data
            layer_weights[indices, :] = 0.
            layer_biases[indices] = 0.
            
            layer.weight.data = layer_weights
            layer.bias.data = layer_biases
        
        return masked_model