import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from argparse import Namespace
from datasets.utils.incremental_dataset import IncrementalDataset, getfeature_loader, get_feature_extractor
from datasets.transforms.denormalization import DeNormalize

class MyImagenet(ImageFolder):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, transform: transforms=None,
                target_transform: transforms=None) -> None:
        super(MyImagenet, self).__init__(
            root, transform)
        self.attributes = []
        self.trans = []
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data = [s[0] for s in self.samples]

    def set_att(self, att_name, att_data, att_transform=None):
        self.attributes.append(att_name)
        self.trans.append(att_transform)
        setattr(self, att_name, att_data)

    def get_att_names(self):
        return self.attributes

    def __getitem__(self, index):
        path, target = self.data[index], self.targets[index]

        # path, target = self.data[index]
        img = self.loader(path)
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = (img, target)
        for i, att in enumerate(self.attributes):
            att_data = getattr(self, att)[index]

            trans = self.trans[i]
            if trans:
                att_data = trans(att_data)

            ret_tuple += (att_data,)

        return ret_tuple

    def __len__(self) -> int:
        return len(self.data)

class SequentialImagenet(IncrementalDataset):
    NAME = 'seq-imagenet'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 100
    N_TASKS = 10
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.nc = 1000
        self.nt = 10
        self.n_channel = 3
        super(SequentialImagenet, self).__init__(args)

        if self.args.featureNet:
            self.args.transform = 'pytorch'
            self.extractor = get_feature_extractor(args)

        if self.args.transform == 'pytorch':
            self.n_imsize1 = 224
            self.n_imsize2 = 224
            self.normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.dnormalization_transform = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                self.normalization_transform])
            self.test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalization_transform])
        else:
            self.n_imsize1 = 64
            self.n_imsize2 = 64
            self.train_transform = transforms.Compose([
                transforms.Resize((68, 68)),
                transforms.CenterCrop((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize((68, 68)),
                transforms.CenterCrop((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def get_data_loaders(self):
        train_dataset = MyImagenet(self.args.root + '/imagenet/' + 'train', transform=self.train_transform)
        test_dataset = MyImagenet(self.args.root + '/imagenet/' + 'val', transform=self.test_transform)

        train, test = getfeature_loader(train_dataset, test_dataset, setting=self)
        return train, test

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.train_transform])
        return transform




