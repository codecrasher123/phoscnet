import torch
import torch.nn as nn

from .pyramidpooling import TemporalPyramidPooling

# Robust timm import: prefer the new path, fall back to the old
try:
    from timm.models import register_model          # newer timm
except Exception:  # ImportError on older layouts
    from timm.models.registry import register_model  # older path 

__all__ = ["PHOSCnet_temporalpooling"]


class PHOSCnet(nn.Module):
    """
    Input: (B, 3, 50, 250)
    Conv → 512 ch → TemporalPyramidPooling([1,2,5]) → 512*(1+2+5)=4096
    Heads:
      - PHOS: 165-dim (regression; ReLU)
      - PHOC: 604-dim (multi-label; Sigmoid)
    """
    def __init__(self):
        super().__init__()

        # VGG-ish feature stack ending at 512 channels
        self.conv = nn.Sequential(
            # 64 @ 50x250
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 64 @ 25x125

            # 128 @ 25x125
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> 128 @ ~12x62

            # 256 block (no further pooling)
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),

            # 512 block
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )

        
        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])
        fc_in = self.temporal_pool.get_output_size(512)  # 512*(1+2+5)=4096

        def mlp():
            return nn.Sequential(
                nn.Linear(fc_in, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(4096, 4096),  nn.ReLU(inplace=True), nn.Dropout(0.5),
            )

        self.phos = nn.Sequential(mlp(), nn.Linear(4096, 165), nn.ReLU(inplace=True))
        self.phoc = nn.Sequential(mlp(), nn.Linear(4096, 604), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)
        return {"phos": self.phos(x), "phoc": self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()
