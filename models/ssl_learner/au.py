import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms as T

from models.ssl_learner.byol import RandomApply


class AU(nn.Module):
    def __init__(self, args, backbone, img_size):
        super().__init__()
        self.args = args
        self.target_encoder = backbone

        self.augmentation = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p=0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p=0.2
            ),
            T.RandomResizedCrop((img_size, img_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])
            ),
        )

    def forward(self, input):
        aug_1, aug_2 = self.augmentation(input), self.augmentation(input)
        feat_1, feat_2 = self.target_encoder.features(aug_1), self.target_encoder.features(aug_2)

        loss = align_loss(feat_1, feat_2) + uniform_loss(torch.cat([feat_1, feat_2]))

        return loss, torch.cat([feat_1, feat_2]), None, torch.cat([aug_1, aug_2])


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
