import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T


class BYOLLearner(nn.Module):
    def __init__(self, args, net, image_size):
        super().__init__()
        self.net = net
        self.args = args

        DEFAULT_AUG = torch.nn.Sequential(
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
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])
            ),
        )

        self.augment1 = DEFAULT_AUG
        self.augment2 = self.augment1

        projection_size, projection_hidden_size = 1024, 512

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=-2).to(
            args.device)

        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size).to(
            args.device)
        self.target_encoder = None

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size, device=args.device))

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def _get_target_encoder(self):
        if self.target_encoder is None:
            target_encoder = copy.deepcopy(self.online_encoder)
            for p in target_encoder.parameters():
                p.requires_grad = False
            self.target_encoder = target_encoder
        return self.target_encoder

    def update_target(self):
        for current_params, ma_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data

            beta = self.args.moving_average_decay
            ma_params.data = beta * old_weight + (1 - beta) * up_weight

    def forward(
            self,
            x,
            feat_only=False
    ):
        assert not (self.training and x.shape[
            0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        target_encoder = self._get_target_encoder()

        if feat_only:
            return self.target_encoder(x, feat_only=True)

        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, online_feat_one = self.online_encoder(image_one)
        online_proj_two, online_feat_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            self.update_target()

            target_proj_one, target_feat_one = target_encoder(image_one)
            target_proj_two, target_feat_two = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = self.loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = self.loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean(), \
               torch.cat([online_feat_one, online_feat_two]), torch.cat([target_feat_one, target_feat_two]), \
               torch.cat([image_one, image_two])


def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


class Classifier_Linear(nn.Module):
    def __init__(self, dim, nc):
        super().__init__()
        self.linear = nn.Linear(dim, nc)

    def forward(self, x):
        y = self.linear(x)
        return y


class Classifier_Square(nn.Module):
    def __init__(self, dim, nc):
        super().__init__()
        self.linear = nn.Linear(dim, nc, bias=False)
        self.c = nn.Parameter(torch.ones(nc), requires_grad=True)

    def forward(self, x):
        y = self.linear(x)
        y = (y - self.c) ** 2
        return y


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _get_projector(self, hidden):
        if self.projector is None:
            # 单例的原因：需要一次 forward pass 来得到特征的维度，从而构建 projector
            _, dim = hidden.shape
            create_mlp_fn = MLP
            projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size)
            self.projector = projector.to(hidden)

        return self.projector

    def forward(self, x, feat_only=False):
        representation = self.net.features(x)

        if feat_only:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation