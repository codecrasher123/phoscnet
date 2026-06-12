import torch
import torch.nn as nn
import torch.nn.functional as F


class PHOSCLoss(nn.Module):
    """
    PyTorch equivalent of the TF setup:
      - PHOS: mean squared error (MSE), weight = 1.5
      - PHOC: binary cross-entropy (BCE on sigmoid outputs), weight = 4.5

    targets: concatenated PHOSC vector per sample [PHOS(165) | PHOC(604)] -> 769
    y: dict with keys:
        "phos": (B, 165)  # regression head (ReLU in the model)
        "phoc": (B, 604)  # multi-label head (Sigmoid in the model)
    """
    def __init__(self, phos_w: float = 1.5, phoc_w: float = 4.5):
        super().__init__()
        self.phos_w = phos_w
        self.phoc_w = phoc_w

    def forward(self, y: dict, targets: torch.Tensor) -> torch.Tensor:
        # Split targets into PHOS and PHOC parts
        t_phos = targets[:, :165]     # (B, 165)
        t_phoc = targets[:, 165:]     # (B, 604)

        # PHOS regression loss (MSE)
        phos_loss = F.mse_loss(y["phos"], t_phos, reduction="mean")

        # PHOC multi-label loss (BCE on probabilities; model already applies Sigmoid)
        phoc_loss = F.binary_cross_entropy(y["phoc"], t_phoc, reduction="mean")

        # Weighted sum (match TF loss_weights: phos=1.5, phoc=4.5)
        return self.phos_w * phos_loss + self.phoc_w * phoc_loss
