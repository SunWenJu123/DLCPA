import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ssl_learner.au import AU
from torch.optim.lr_scheduler import StepLR
from models.ssl_learner.byol import BYOLLearner
from models.ssl_learner.moco import MoCoLearner

from backbone.MNISTMLP import MNISTMLP
from backbone.ResNet import resnet18, resnet34
from models.ssl_learner.simclr import SimClrLearner
from models.ssl_learner.vicreg import VICReg
from models.utils.incremental_model import IncrementalModel


class DLCPA(IncrementalModel):
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, args):
        super(DLCPA, self).__init__(args)
        self.epochs = args.n_epochs
        self.net, self.net_old, self.classifier = None, None, None
        self.loss = F.cross_entropy

        self.current_task = -1

    def begin_il(self, dataset):

        if self.args.dataset == 'seq-mnist':
            self.net = MNISTMLP(28 * 28, dataset.nc).to(self.device)
        else:
            if self.args.featureNet:
                self.net = MNISTMLP(1000, dataset.nc, hidden_dim=[800, 500]).to(self.device)
            elif self.args.backbone == 'None' or self.args.backbone == 'resnet18':
                self.net = resnet18(dataset.nc).to(self.device)
            elif self.args.backbone == 'resnet34':
                self.net = resnet34(dataset.nc).to(self.device)

        self.net_old = copy.deepcopy(self.net)


        self.latent_dim = self.net.nf * 8
        self.classifier = Classifier_Linear(self.latent_dim, dataset.nc).to(self.device)

        img_size = dataset.n_imsize1
        if self.args.ssl_leaner == 'byol':
            self.learner = BYOLLearner(self.args, copy.deepcopy(self.net), img_size).to(self.device)
        elif self.args.ssl_leaner == 'moco':
            self.learner = MoCoLearner(self.args, copy.deepcopy(self.net), img_size).to(self.device)
        elif self.args.ssl_leaner == 'vicreg':
            self.learner = VICReg(self.args, copy.deepcopy(self.net), img_size).to(self.device)
        elif self.args.ssl_leaner == 'au':
            self.learner = AU(self.args, copy.deepcopy(self.net), img_size).to(self.device)
        elif self.args.ssl_leaner == 'simclr':
            self.learner = SimClrLearner(self.args, copy.deepcopy(self.net), img_size, self.latent_dim).to(self.device)


        self.cpt = int(dataset.nc / dataset.nt)
        self.t_c_arr = dataset.t_c_arr
        self.eye = torch.tril(torch.ones((dataset.nc, dataset.nc))).bool().to(self.device)

    def train_task(self, dataset, train_loader):
        self.current_task += 1

        if self.args.ssl_leaner == 'moco':
            self.learner.refresh_queue()

        self.train_(train_loader)

        self.net_old = copy.deepcopy(self.net)


    def train_(self, train_loader):
        cur_class = self.t_c_arr[self.current_task]
        print('learning classes: ', cur_class)

        # self.net.train()
        opt_learner = torch.optim.SGD(self.learner.parameters(), lr=self.args.lr)
        opt_classifier = torch.optim.SGD(self.classifier.parameters(), lr=self.args.clslr)

        scheduler_feature = StepLR(opt_learner, step_size=self.args.scheduler_step, gamma=0.1)
        scheduler_classifier = StepLR(opt_classifier, step_size=self.args.scheduler_step, gamma=0.1)
        for epoch in range(self.epochs):
            for step, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                l_loss, online_feat, target_feat, inputs = self.learner(inputs)
                if self.args.ssl_leaner == 'byol' or self.args.ssl_leaner == 'vicreg' or self.args.ssl_leaner == 'au' or self.args.ssl_leaner == 'simclr':
                    labels = torch.cat([labels, labels])

                # =====  =====  ===== update learner =====  =====  =====
                pred = self.classifier(online_feat)
                supervised_loss = self.args.ssl_weight * l_loss + self.args.sl_weight * self.loss(
                    pred[:, cur_class[0]: cur_class[-1] + 1],
                    labels - cur_class[0]
                )

                opt_learner.zero_grad()
                supervised_loss.backward(retain_graph=False)
                opt_learner.step()


                # # =====  =====  ===== update target network =====  =====  =====
                for online_params, target_params, old_params \
                        in zip(self.learner.target_encoder.parameters(), self.net.parameters(),
                               self.net_old.parameters()):
                    online_weight, target_weight, old_weight = online_params.data, target_params.data, old_params.data
                    target_params.data = (self.current_task * old_weight + online_weight * 1.) / (self.current_task + 1)

                # =====  =====  ===== update classifier =====  =====  =====
                with torch.no_grad():
                    feat = self.net.features(inputs)

                pred = self.classifier(feat)
                loss_ce = self.loss(
                    pred[:, cur_class],
                    labels - cur_class[0]
                )
                classifier_loss = loss_ce

                opt_classifier.zero_grad()
                classifier_loss.backward()
                opt_classifier.step()

            scheduler_feature.step()
            scheduler_classifier.step()
            if epoch % self.args.print_freq == 0:
                print('epoch:%d, feat_extract_loss:%.5f, classifier_loss:%.5f' % (
                    epoch, supervised_loss.to('cpu').item(), classifier_loss.to('cpu').item()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur_class = self.t_c_arr[self.current_task]
        # self.net.eval()
        x = x.to(self.device)
        with torch.no_grad():
            feat = self.net.features(x)
            outputs = self.classifier(feat)
            # outputs = outputs[:, :cur_class[-1] + 1]
        return outputs

    def test_task(self, dataset, test_loader):
        pass


class Classifier_Linear(nn.Module):
    def __init__(self, dim, nc):
        super().__init__()
        self.linear = nn.Linear(dim, nc)

    def forward(self, x):
        y = self.linear(x)
        return y
