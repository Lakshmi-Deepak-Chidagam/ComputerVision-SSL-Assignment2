"""Exact ResNet18 and SimCLR definitions from SSL_Assignment.ipynb."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

def create_model(device="cpu"):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 16)
    return model.to(device)

class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        features = self.encoder(x)
        return F.normalize(self.projector(features), dim=1)

def nt_xent(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    logits = z @ z.T / temperature
    logits.fill_diagonal_(-1e9)

    targets = (
        torch.arange(2 * batch_size, device=z.device)
        + batch_size
    ) % (2 * batch_size)

    return F.cross_entropy(logits, targets)
