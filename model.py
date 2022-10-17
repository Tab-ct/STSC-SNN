import torch
import torch.nn as nn
from spikingjelly.activation_based import layer
from spikingjelly.activation_based import surrogate, neuron
import modules

class SHD_STSC(nn.Module):
    def __init__(self):
        super().__init__()

        time_rf_conv = 5
        time_rf_at = 3

        self.fc = nn.Sequential(
            modules.STSC(700,dimension=2,time_rf_conv=time_rf_conv, time_rf_at=time_rf_at, use_gate=True, use_filter=True),
            layer.Linear(700,128),
            neuron.LIFNode(tau=10.0, decay_input=False, v_threshold=0.3, surrogate_function=surrogate.ATan(), detach_reset=True),
            # modules.STSC(128,dimension=2,time_rf_conv=time_rf_conv, time_rf_at=time_rf_at, use_gate=True, use_filter=True),
            layer.Linear(128,128),
            neuron.LIFNode(tau=10.0, decay_input=False, v_threshold=0.3, surrogate_function=surrogate.ATan(), detach_reset=True),
            # modules.STSC(128,dimension=2,time_rf_conv=time_rf_conv, time_rf_at=time_rf_at, use_gate=True, use_filter=True),
            layer.Linear(128,100),
            neuron.LIFNode(tau=10.0, decay_input=False, v_threshold=0.3, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.VotingLayer(5)
        )

    def forward(self, x: torch.Tensor):
        return self.fc(x)





    