
from torchvision import transforms
import torch
from PIL import Image

class autoaug(object) :
    def __init__(self, dataset):

        if dataset == "cifar" :
            dataset = "cifar10"
        elif dataset == "svhn" :
            dataset = "svhn"

        policy = transforms.AutoAugmentPolicy(dataset);

        transform_train = [
            transforms.AutoAugment(policy=policy),
            transforms.ToTensor(),
        ]

        self.transform = transforms.Compose(transform_train)

        self.meta_aug = True

    def fit(self, data, label):
        pass

    def __call__(self, img1, idx, idx2) : #

        img_pil = Image.fromarray((img1 * 255.0).
                        permute(1, 2, 0).type(torch.uint8).data.numpy()).convert("RGB");

        output = self.transform(img_pil);

        return output


class randaug(object) :
    def __init__(self, dataset):

        transform_train = [
            transforms.RandAugment(),
            transforms.ToTensor(),
        ]

        self.transform = transforms.Compose(transform_train)

        self.meta_aug = True

    def fit(self, data, label):
        pass

    def __call__(self, img1, idx, idx2) : #

        img_pil = Image.fromarray((img1 * 255.0).
                        permute(1, 2, 0).type(torch.uint8).data.numpy()).convert("RGB");

        output = self.transform(img_pil);

        return output


class trivialaug(object) :
    def __init__(self, dataset):

        transform_train = [
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
        ]

        self.transform = transforms.Compose(transform_train)

        self.meta_aug = True

    def fit(self, data, label):
        pass

    def __call__(self, img1, idx, idx2) : #

        img_pil = Image.fromarray((img1 * 255.0).
                        permute(1, 2, 0).type(torch.uint8).data.numpy()).convert("RGB");

        output = self.transform(img_pil);

        return output


