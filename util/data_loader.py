
from torch.utils.data import DataLoader

from torchvision.datasets.folder import default_loader
import numpy as np
import os

import torch.nn.functional as F

import random
from torchvision import transforms
import torch.utils.data as data
import torch
from torch_deform_conv.deform_conv import th_batch_map_offsets, th_generate_grid
import torchvision.datasets.cifar as cifar

import torchvision.datasets.mnist as MNIST
import torchvision.datasets.svhn as SVHN

import pandas as pd
from PIL import Image


def gen_one_hot_np(num, class_idx, tot_class=10) :

    one_hot_label = np.zeros(shape=(num, tot_class));
    one_hot_label[:, class_idx] = 1.0;

    return one_hot_label


def gen_one_hot_tensor(num, class_idx, class_idx2=None, lam=None, tot_class=10, prob=1.0, gpu=False):

    if class_idx2 is None :

        target_label = torch.zeros(size=(num, tot_class));
        target_label[:,:] = (1-prob)/(tot_class-1);
        target_label[torch.arange(target_label.size(0)), class_idx] = prob

    else :
        target_label = torch.zeros(size=(num, tot_class));
        target_label[torch.arange(target_label.size(0)), class_idx] = lam
        target_label[torch.arange(target_label.size(0)), class_idx2] = (1-lam)

    if gpu :
        target_label = target_label.cuda()
    return target_label


class CustomAugDataSet(data.Dataset) :

    def __init__(self, data, transform=None, seed=None):
        self.transform = transform
        self.data = data[0];
        self.targets = data[1];
        self.attentions = data[2];

        self.ori_data = self.data.clone();
        self.ori_targets = self.targets.clone();

        self.seed = seed;

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)


    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(2), x.size(3)
        dtype, cuda = x.data.type(), x.data.is_cuda
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, c, h, w) -> (b*c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))
        return x

    @staticmethod
    def _to_b_c_h_w(x, x_shape):
        """(b*c, h, w) -> (b, c, h, w)"""
        x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
        return x

    def __getitem__(self, index):

        img, target, attention = self.data[index], self.targets[index], self.attentions[index]

        ratio = np.random.random()
        output = (img * attention[:1] + img * ratio * attention[1:]);

        '''
        rand_noise = torch.rand(size=(1,)).to(offset.device)/2.0 + 0.5;
        rand_offset = torch.rand(size=(1,)).to(offset.device)/2.0 + 0.5;

        noise_aug = rand_noise.unsqueeze(-1).unsqueeze(-1) * noise
        noised_img = img + noise_aug;

        offset_aug = rand_offset.unsqueeze(-1).unsqueeze(-1) * offset;


        x_offset, grid = th_batch_map_offsets(noised_img.unsqueeze(0), offset_aug.unsqueeze(0),
                                              grid=self._get_grid(self, img.unsqueeze(0)))
        noised_img = self._to_b_c_h_w(x_offset, img.unsqueeze(0).shape)[0]

        noised_img = torch.clamp(noised_img, min=0.0, max=1.0);
        '''

        if self.transform is not None:
            if np.random.rand() > 0.5 :
                output = self.transform(output)
            else :
                output = output;

        return output, target

    def __len__(self):
        return len(self.data)

class CustomDataSet(data.Dataset) :

    def __init__(self, data, transform=None, seed=None, aug_model=None):
        self.transform = transform
        self.data = data[0];
        self.targets = data[1];
        self.aug_model = aug_model

        if self.aug_model is None :
            self.aug_on = False;
        else :
            self.aug_on = True;

        self.seed = seed;

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)

    def set_aug_on(self):
        if self.aug_model is not None :
            self.aug_on = True;

    def set_aug_off(self):
        if self.aug_model is not None :
            self.aug_on = False;

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]

        same_labels = torch.where(self.targets == self.targets[index])[0];
        same_labels = list(same_labels.cpu().data.numpy());
        same_labels.remove(index)
        index2 = np.random.choice(same_labels);

        img2 = self.data[np.random.choice(same_labels)];

        if self.aug_on:
            if np.random.rand() > 0.5 :
                img = self.aug_model(img, index, index2);

        if self.transform is not None:
            if np.random.rand() > 0.5 :
                img = self.transform(img)
                img2 = self.transform(img2)

        return img, img2, target

    def __len__(self):
        return len(self.targets)


class SyntheticDataSet(data.Dataset) :

    def __init__(self, data, label, eigs, transform=None):
        self.data = data;
        self.transform = transform;
        self.labels = label;
        self.eigs = eigs;


    def __getitem__(self, index):

        '''
        img, label, eig, m_eigs = self.data[index], self.labels[index], \
                          self.eigs[index], self.mean_eigs[index];

        return img, label, eig, m_eigs
        '''

        img, label, eig = self.data[index], self.labels[index], self.eigs[index]


        if self.transform is not None:
            if np.random.rand() > 0.5 :
                img = self.transform(img)

        return img, label, eig

    def __len__(self):
        return len(self.data)



def load_data_CUB200(batch_size, train_size=8, val_size=10, seed=25, aug_model=None, num_classes=5) :

    np.random.seed(seed);
    torch.manual_seed(seed);

    transform_train = transforms.Compose([
        transforms.RandomCrop(84),
        transforms.ColorJitter(brightness=0.4, contrast=(0.5, 1.5)),
        transforms.RandomHorizontalFlip(),
    ])

    images = pd.read_csv(os.path.join("./dataset", 'CUB_200_2011', 'images.txt'), sep=' ',
                         names=['img_id', 'filepath'])
    image_class_labels = pd.read_csv(os.path.join("./dataset", 'CUB_200_2011', 'image_class_labels.txt'),
                                     sep=' ', names=['img_id', 'target'])
    train_test_split = pd.read_csv(os.path.join("./dataset", 'CUB_200_2011', 'train_test_split.txt'),
                                   sep=' ', names=['img_id', 'is_training_img'])

    data = images.merge(image_class_labels, on='img_id')
    data = data.merge(train_test_split, on='img_id')


    trainset = data[data.is_training_img == 1]
    valset = data[data.is_training_img == 0]

    train_x_list = list();
    train_y_list = list();

    val_x_list = list();
    val_y_list = list();

    classes = np.random.choice(np.unique(valset.target), replace=False, size=num_classes);

    for c_idx, c in enumerate(classes):
        np.random.seed(seed);
        torch.manual_seed(seed);

        idx_c = np.where(trainset.target == c)[0];
        idx_c = np.random.permutation(idx_c);

        for idx in idx_c[:train_size] :
            sample = trainset.iloc[idx]
            path = os.path.join("./dataset", "CUB_200_2011/images", sample.filepath);
            img = default_loader(path);
            img = img.resize((84,84));
            '''
            import cv2
            cv2.imshow("test", np.asarray(img)[:,:,(2,1,0)])
            cv2.waitKey(0)
            '''
            rescaled_train = torch.FloatTensor(np.asarray(img)) / 255.0;
            rescaled_train = rescaled_train.unsqueeze(0).permute(0, 3, 1, 2);
            train_x_list.extend(rescaled_train);

        idx_c = np.where(valset.target == c)[0];
        idx_c = np.random.permutation(idx_c);

        for idx in idx_c[:val_size] :
            sample = valset.iloc[idx]
            path = os.path.join("./dataset", "CUB_200_2011/images", sample.filepath);
            img = default_loader(path);
            img = img.resize((84, 84));

            rescaled_val = torch.FloatTensor(np.asarray(img)) / 255.0;
            rescaled_val = rescaled_val.unsqueeze(0).permute(0, 3, 1, 2);
            val_x_list.extend(rescaled_val);

        # np.tile(np.ones(len(train_img_list[c_count]), dtype=np.int32) * c_count
        train_y_list.extend(gen_one_hot_tensor(train_size, c_idx, tot_class=num_classes));
        val_y_list.extend(gen_one_hot_tensor(len(idx_c[:val_size]), c_idx, tot_class=num_classes));

    train_x = torch.stack(train_x_list);
    train_y = torch.argmax(torch.stack(train_y_list), 1);

    val_x = torch.stack(val_x_list);
    val_y = torch.argmax(torch.stack(val_y_list), 1)

    if aug_model is not None:
        aug_model.fit(train_x, train_y);


    train_data = CustomDataSet((train_x, train_y), seed=seed, transform=transform_train, aug_model=aug_model)

    valid_data = CustomDataSet((val_x, val_y), seed=seed)

    train_data_no_augment = CustomDataSet((train_x, train_y), seed=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=max(batch_size, 16))

    train_loader_no_augment = DataLoader(train_data_no_augment, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, train_loader_no_augment, valid_loader, train_x, train_y, transform_train


def load_data_cifar100(batch_size, train_size=8, val_size=10, seed=25, aug_model=None, num_classes=5) :

    hierarchy = {
    'aquatic_mammals':['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish':	['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers':['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers' : ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables':['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices' :['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture':	['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects':	['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores':['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man-made_outdoor_things':['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes':['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores' :	['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates':	['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people':	['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees' :	['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1':['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    }

    super_classes = ["aquatic_mammals", "insects", "medium_mammals", "people"];

    np.random.seed(seed);
    torch.manual_seed(seed);

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    trainset = cifar.CIFAR100(root='./dataset', train=True, download=True)
    valset = cifar.CIFAR100(root='./dataset', train=False, download=True)

    train_img = trainset.data;
    train_label = np.stack(trainset.targets);

    val_img = valset.data
    val_label = np.stack(valset.targets);

    train_x_list = list();
    train_y_list = list();

    val_x_list = list();
    val_y_list = list();

    possible_classes = [];
    possible_class_idxes = [];

    [possible_classes.extend(hierarchy[coarse_cls]) for coarse_cls in super_classes];

    all_class = valset.classes;
    for c_idx, cls in enumerate(all_class) :
        if cls in possible_classes :
            possible_class_idxes.append(c_idx);

    classes = np.random.choice(possible_class_idxes, replace=False, size=5);

    print(classes)
    for c_idx, c in enumerate(classes):
        np.random.seed(seed);
        torch.manual_seed(seed);

        idx_c = np.where(train_label == c)[0];
        idx_c = np.random.permutation(idx_c);

        rescaled_train = torch.FloatTensor(train_img[idx_c[:train_size]]) / 255.0;
        rescaled_train = rescaled_train.permute(0, 3, 1, 2);

        idx_c = np.where(val_label == c)[0];
        # np.random.shuffle(idx_c);
        idx_c = np.random.permutation(idx_c);

        rescaled_val = torch.FloatTensor(val_img[idx_c[:val_size]]) / 255.0;
        rescaled_val = rescaled_val.permute(0, 3, 1, 2);

        train_x_list.extend(rescaled_train);
        val_x_list.extend(rescaled_val);

        # np.tile(np.ones(len(train_img_list[c_count]), dtype=np.int32) * c_count
        train_y_list.extend(gen_one_hot_tensor(train_size, c_idx, tot_class=num_classes));
        val_y_list.extend(gen_one_hot_tensor(val_size, c_idx, tot_class=num_classes));

    train_x = torch.stack(train_x_list);
    train_y = torch.argmax(torch.stack(train_y_list), 1);

    val_x = torch.stack(val_x_list);
    val_y = torch.argmax(torch.stack(val_y_list), 1)

    if aug_model is not None:
        aug_model.fit(train_x, train_y);


    train_data = CustomDataSet((train_x, train_y), seed=seed, transform=transform_train, aug_model=aug_model)

    valid_data = CustomDataSet((val_x, val_y), seed=seed)

    train_data_no_augment = CustomDataSet((train_x, train_y), seed=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=max(batch_size, 128))

    train_loader_no_augment = DataLoader(train_data_no_augment, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, train_loader_no_augment, valid_loader, train_x, train_y, transform_train

def load_data_fmnist(batch_size, train_size = 8, val_size = 10, seed = 25) :

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
    ])

    trainset = MNIST.FashionMNIST(root='./dataset',  train=True, download=True);
    testset = MNIST.FashionMNIST(root='./dataset', train=False, download=True);

    tot_train_data = trainset.train_data
    tot_train_label = trainset.targets

    tot_val_data = testset.test_data
    tot_val_label = testset.targets

    train_x_list = list();
    train_y_list = list();

    val_x_list = list();
    val_y_list = list();

    for c_idx in range(10) :
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_label_idx = (tot_train_label == c_idx).nonzero();
        random_idx = torch.randperm(train_label_idx.shape[0])

        train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

        train_x = F.interpolate(tot_train_data[train_label_idx].unsqueeze(1),
                                size=[32,32]).float()/255.0;
        train_y = tot_train_label[train_label_idx];

        val_label_idx = (tot_val_label == c_idx).nonzero();
        random_idx = torch.randperm(val_label_idx.shape[0])
        val_label_idx = val_label_idx[random_idx[:val_size]].squeeze();

        val_x = F.interpolate(tot_val_data[val_label_idx].unsqueeze(1),
                              size=[32,32]).float()/255.0;
        val_y = tot_val_label[val_label_idx];

        train_x_list.extend(train_x);
        train_y_list.extend(gen_one_hot_tensor(train_y.shape[0], c_idx));
        val_x_list.extend(val_x);
        val_y_list.extend(gen_one_hot_tensor(val_y.shape[0], c_idx));


    train_x = torch.stack(train_x_list);
    train_y = torch.argmax(torch.stack(train_y_list), 1);

    val_x = torch.stack(val_x_list);
    val_y = torch.argmax(torch.stack(val_y_list),1)

    train_data = CustomDataSet((train_x, train_y), seed=seed, transform=transform_train)
    valid_data = CustomDataSet((val_x, val_y), seed=seed)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=max(batch_size,128))

    return train_data_loader, valid_data_loader, train_x, train_y, transform_train

def _get_grid(x):
    batch_size, input_height, input_width = x.size(0), x.size(2), x.size(3)
    dtype, cuda = x.data.type(), x.data.is_cuda
    _grid_param = (batch_size, input_height, input_width, dtype, cuda)
    _grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
    return _grid

def _to_b_c_h_w(x, x_shape):
    """(b*c, h, w) -> (b, c, h, w)"""
    x = x.contiguous().view(-1, int(x_shape[1]), int(x_shape[2]), int(x_shape[3]))
    return x


def load_data_m_mnist(batch_size, train_size = 8, val_size = 10, seed = 25, aug_model=None) :
    transform_train = transforms.Compose([
    ])

    trainset = MNIST.MNIST(root='./dataset', train=True, download=True);
    testset = MNIST.MNIST(root='./dataset', train=False, download=True);

    tot_train_data = trainset.train_data
    tot_train_label = trainset.targets

    tot_val_data = testset.test_data
    tot_val_label = testset.targets

    train_x_list = list();
    train_y_list = list();

    val_x_list = list();
    val_y_list = list();

    for c_idx in range(10):
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_label_idx = (tot_train_label == c_idx).nonzero();
        random_idx = torch.randperm(train_label_idx.shape[0])

        train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

        train_x = F.interpolate(tot_train_data[train_label_idx].unsqueeze(1),
                                size=[32, 32]).float() / 255.0;

        new_x = torch.empty_like(train_x);

        for idx, x in enumerate(train_x) :
            offset = torch.cat([torch.ones_like(x), torch.ones_like(x)]);
            offset = offset.unsqueeze(0);

            offset = offset.permute(0,2,3,1)

            src_x = train_x[0].unsqueeze(0);

            if idx == 0 :
                offset *= -4.0
            elif idx == 1 :
                offset *= -2.0
            elif idx == 2 :
                offset *= 0.0;
            elif idx == 3 :
                offset *= 2.0
            elif idx == 4 :
                offset *= 4.0

            x_offset, _ = th_batch_map_offsets(src_x, offset,
                                               grid=_get_grid(x.unsqueeze(0)))

            transformed_result = _to_b_c_h_w(x_offset, x.unsqueeze(0).shape)

            new_x[idx] = transformed_result[0]

        train_x_list.extend(new_x);
        train_y = tot_train_label[train_label_idx];

        val_label_idx = (tot_val_label == c_idx).nonzero();
        random_idx = torch.randperm(val_label_idx.shape[0])
        val_label_idx = val_label_idx[random_idx[:val_size]].squeeze();

        val_x = F.interpolate(tot_val_data[val_label_idx].unsqueeze(1),
                              size=[32, 32]).float() / 255.0;
        val_y = tot_val_label[val_label_idx];

        val_x_list.extend(val_x);
        train_y_list.extend(gen_one_hot_tensor(train_y.shape[0], c_idx));

        val_y_list.extend(gen_one_hot_tensor(val_y.shape[0], c_idx));

    train_x = torch.stack(train_x_list);
    train_y = torch.argmax(torch.stack(train_y_list), 1);

    val_x = torch.stack(val_x_list);
    val_y = torch.argmax(torch.stack(val_y_list), 1)

    if aug_model is not None:
        aug_model.fit(train_x, train_y);


    train_data = CustomDataSet((train_x, train_y), seed=seed, transform=transform_train, aug_model=aug_model)

    valid_data = CustomDataSet((val_x, val_y), seed=seed)
    train_data_no_augment = CustomDataSet((train_x, train_y), seed=seed)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # aug_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_data_loader = DataLoader(valid_data, batch_size=max(batch_size, 128))

    train_loader_no_augment = DataLoader(train_data_no_augment, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_data_loader, train_loader_no_augment, valid_data_loader, train_x, train_y, transform_train


def load_data_c_mnist(batch_size, train_size = 8, val_size = 10, seed = 25, aug_model=None) :
    transform_train = transforms.Compose([
    ])

    colors = [(255,0,0), (0,255,0), (0,0,255)];

    trainset = MNIST.MNIST(root='./dataset', train=True, download=True);
    testset = MNIST.MNIST(root='./dataset', train=False, download=True);

    tot_train_data = trainset.train_data
    tot_train_label = trainset.targets

    tot_val_data = testset.test_data
    tot_val_label = testset.targets

    train_x_list = list();
    train_y_list = list();

    val_x_list = list();
    val_y_list = list();

    for c_idx in range(10):
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_label_idx = (tot_train_label == c_idx).nonzero();
        random_idx = torch.randperm(train_label_idx.shape[0])

        train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

        train_x = F.interpolate(tot_train_data[train_label_idx].unsqueeze(1),
                                size=[32, 32]).float() / 255.0;

        '''
        new_x = torch.empty_like(train_x);
        
        for idx, x in enumerate(train_x) :
            offset = torch.cat([torch.ones_like(x), torch.ones_like(x)]);
            offset = offset.unsqueeze(0);

            offset = offset.permute(0,2,3,1)

            src_x = train_x[0].unsqueeze(0);

            if idx == 0 :
                offset *= -3.0
                x_offset, _ = th_batch_map_offsets(src_x, offset,
                                                   grid=_get_grid(x.unsqueeze(0)))

                transformed_result = _to_b_c_h_w(x_offset, x.unsqueeze(0).shape)

            elif idx == 1 :
                offset *= -0.0
                x_offset, _ = th_batch_map_offsets(src_x, offset,
                                                   grid=_get_grid(x.unsqueeze(0)))

                transformed_result = _to_b_c_h_w(x_offset, x.unsqueeze(0).shape)

            elif idx == 2 :
                offset *= 3.0
                x_offset, _ = th_batch_map_offsets(src_x, offset,
                                                   grid=_get_grid(x.unsqueeze(0)))

                transformed_result = _to_b_c_h_w(x_offset, x.unsqueeze(0).shape)

            new_x[idx] = transformed_result[0]
            #new_x =

        train_x = new_x;
        '''
        train_y = tot_train_label[train_label_idx];

        val_label_idx = (tot_val_label == c_idx).nonzero();
        random_idx = torch.randperm(val_label_idx.shape[0])
        val_label_idx = val_label_idx[random_idx[:val_size]].squeeze();

        val_x = F.interpolate(tot_val_data[val_label_idx].unsqueeze(1),
                              size=[32, 32]).float() / 255.0;
        val_y = tot_val_label[val_label_idx];
        new_x = torch.empty(size=(train_size,3,) + train_x.shape[-2:]);

        for idx, x in enumerate(train_x) :

            if idx == 0 :
                colored_x = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)]);
            elif idx == 1 :
                colored_x = torch.cat([x*0.5, x*0.5, torch.zeros_like(x)]);
            elif idx == 2 :
                colored_x = torch.cat([torch.zeros_like(x), x, torch.zeros_like(x)]);
            elif idx == 3 :
                colored_x = torch.cat([torch.zeros_like(x), x*0.5, x*0.5]);
            elif idx == 4 :
                colored_x = torch.cat([torch.zeros_like(x), torch.zeros_like(x), x]);
            elif idx == 5 :
                colored_x = torch.cat([x*0.5, torch.zeros_like(x), x*0.5]);

            new_x[idx] = colored_x

        train_x_list.extend(new_x);

        new_x = torch.empty(size=(val_size,3,) + train_x.shape[-2:]);
        for idx, x in enumerate(val_x) :

            rand3 = np.random.random(3);
            ratio = rand3/(np.sum(rand3));

            colored_x = torch.cat([x*ratio[0], x*ratio[1], x*ratio[2]]);

            new_x[idx] = colored_x

        val_x_list.extend(new_x);
        train_y_list.extend(gen_one_hot_tensor(train_y.shape[0], c_idx));

        val_y_list.extend(gen_one_hot_tensor(val_y.shape[0], c_idx));

    train_x = torch.stack(train_x_list);
    train_y = torch.argmax(torch.stack(train_y_list), 1);

    val_x = torch.stack(val_x_list);
    val_y = torch.argmax(torch.stack(val_y_list), 1)

    if aug_model is not None:
        aug_model.fit(train_x, train_y);


    train_data = CustomDataSet((train_x, train_y), seed=seed, transform=transform_train, aug_model=aug_model)

    valid_data = CustomDataSet((val_x, val_y), seed=seed)
    train_data_no_augment = CustomDataSet((train_x, train_y), seed=seed)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # aug_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_data_loader = DataLoader(valid_data, batch_size=max(batch_size, 128))

    train_loader_no_augment = DataLoader(train_data_no_augment, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_data_loader, train_loader_no_augment, valid_data_loader, train_x, train_y, transform_train


def load_data_mnist(batch_size, train_size = 8, val_size = 10, seed = 25, aug_model=None) :

    transform_train = transforms.Compose([
    ])


    trainset = MNIST.MNIST(root='./dataset',  train=True, download=True);
    testset = MNIST.MNIST(root='./dataset', train=False, download=True);

    tot_train_data = trainset.train_data
    tot_train_label = trainset.targets

    tot_val_data = testset.test_data
    tot_val_label = testset.targets

    train_x_list = list();
    train_y_list = list();

    val_x_list = list();
    val_y_list = list();

    for c_idx in range(10) :
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_label_idx = (tot_train_label == c_idx).nonzero();
        random_idx = torch.randperm(train_label_idx.shape[0])

        train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

        train_x = F.interpolate(tot_train_data[train_label_idx].unsqueeze(1),
                                size=[32,32]).float()/255.0;
        train_y = tot_train_label[train_label_idx];

        val_label_idx = (tot_val_label == c_idx).nonzero();
        random_idx = torch.randperm(val_label_idx.shape[0])
        val_label_idx = val_label_idx[random_idx[:val_size]].squeeze();

        val_x = F.interpolate(tot_val_data[val_label_idx].unsqueeze(1),
                              size=[32,32]).float()/255.0;
        val_y = tot_val_label[val_label_idx];

        train_x_list.extend(train_x);
        train_y_list.extend(gen_one_hot_tensor(train_y.shape[0], c_idx));
        val_x_list.extend(val_x);
        val_y_list.extend(gen_one_hot_tensor(val_y.shape[0], c_idx));


    train_x = torch.stack(train_x_list);
    train_y = torch.argmax(torch.stack(train_y_list), 1);

    val_x = torch.stack(val_x_list);
    val_y = torch.argmax(torch.stack(val_y_list),1)

    if aug_model is not None:
        aug_model.fit(train_x, train_y);


    train_data = CustomDataSet((train_x, train_y), seed=seed, transform=transform_train, aug_model=aug_model)

    valid_data = CustomDataSet((val_x, val_y), seed=seed)
    train_data_no_augment = CustomDataSet((train_x, train_y), seed=seed)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    # aug_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_data_loader = DataLoader(valid_data, batch_size=max(batch_size, 128))

    train_loader_no_augment = DataLoader(train_data_no_augment, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_data_loader, train_loader_no_augment, valid_data_loader, train_x, train_y, transform_train


def load_data_svhn(batch_size, train_size = 8, val_size = 10, seed = 25, aug_model=None) :

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=(0.2, 2.0)),
        transforms.RandomCrop(32, padding=4),
    ])

    trainset = SVHN.SVHN(root='./dataset',  download=True, split="train");
    testset = SVHN.SVHN(root='./dataset', download=True, split="test");

    tot_train_data = trainset.data
    tot_train_label = trainset.labels

    tot_val_data = testset.data
    tot_val_label = testset.labels

    train_x_list = list();
    train_y_list = list();

    val_x_list = list();
    val_y_list = list();

    for c_idx in range(10) :
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_label_idx = (tot_train_label == c_idx).nonzero()[0];
        random_idx = torch.randperm(train_label_idx.shape[0])
        train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

        train_x = tot_train_data[train_label_idx]/255.0;

        train_y = tot_train_label[train_label_idx];

        val_label_idx = (tot_val_label == c_idx).nonzero()[0];
        random_idx = torch.randperm(val_label_idx.shape[0])
        val_label_idx = val_label_idx[random_idx[:val_size]].squeeze();

        val_x = tot_val_data[val_label_idx]/255.0;
        val_y = tot_val_label[val_label_idx];

        train_x_list.extend(torch.FloatTensor(train_x));
        train_y_list.extend(gen_one_hot_tensor(train_y.shape[0], c_idx));
        val_x_list.extend(torch.FloatTensor(val_x));
        val_y_list.extend(gen_one_hot_tensor(val_y.shape[0], c_idx));


    train_x = torch.stack(train_x_list);
    train_y = torch.argmax(torch.stack(train_y_list), 1);

    val_x = torch.stack(val_x_list);
    val_y = torch.argmax(torch.stack(val_y_list),1)

    if aug_model is not None :
        aug_model.fit(train_x, train_y);

    train_data = CustomDataSet((train_x, train_y), seed=seed, transform=transform_train, aug_model=aug_model)
    valid_data = CustomDataSet((val_x, val_y), seed=seed)
    train_data_no_augment = CustomDataSet((train_x, train_y), seed=seed)


    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    #aug_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_data_loader = DataLoader(valid_data, batch_size=max(batch_size,128))

    train_loader_no_augment = DataLoader(train_data_no_augment, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_data_loader, train_loader_no_augment, valid_data_loader, train_x, train_y, transform_train



def sort_by_name(files) :
    file_idxs = [int(file.split("_")[0]) for file in files];

    idxes = np.argsort(file_idxs);

    return [files[idx] for idx in idxes];


def load_data_cifar(batch_size, train_size = 8, val_size = 10, seed = 25, aug_model=None) :

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    trainset = cifar.CIFAR10(root='./dataset', train=True, download=True)
    valset = cifar.CIFAR10(root='./dataset', train=False, download=True)

    train_img = trainset.data;
    train_label = np.stack(trainset.targets);

    val_img = valset.data
    val_label = np.stack(valset.targets);

    train_x_list = list();
    train_y_list = list();

    val_x_list = list();
    val_y_list = list();

    for c in range(10):
        np.random.seed(seed);
        torch.manual_seed(seed);

        idx_c = np.where(train_label == c)[0];
        idx_c = np.random.permutation(idx_c);

        rescaled_train = torch.FloatTensor(train_img[idx_c[:train_size]]) / 255.0;
        rescaled_train = rescaled_train.permute(0,3,1,2);

        idx_c = np.where(val_label == c)[0];
        # np.random.shuffle(idx_c);
        idx_c = np.random.permutation(idx_c);

        rescaled_val = torch.FloatTensor(val_img[idx_c[:val_size]]) / 255.0;
        rescaled_val = rescaled_val.permute(0, 3, 1, 2);

        train_x_list.extend(rescaled_train);
        val_x_list.extend(rescaled_val);

        # np.tile(np.ones(len(train_img_list[c_count]), dtype=np.int32) * c_count
        train_y_list.extend(gen_one_hot_tensor(train_size, c));
        val_y_list.extend(gen_one_hot_tensor(val_size, c));

    train_x = torch.stack(train_x_list);
    train_y = torch.argmax(torch.stack(train_y_list), 1);

    val_x = torch.stack(val_x_list);
    val_y = torch.argmax(torch.stack(val_y_list),1)

    if aug_model is not None :
        outputs = aug_model.fit(train_x, train_y);

        if outputs is not None :

            for idx, output in enumerate(outputs) :
                train_x = torch.cat([train_x, output]);
                train_y = torch.cat([train_y, torch.argmax(gen_one_hot_tensor(len(output), idx),1)])


    train_data = CustomDataSet((train_x, train_y), seed=seed, transform=transform_train, aug_model=aug_model)

    valid_data = CustomDataSet((val_x, val_y), seed=seed)

    train_data_no_augment = CustomDataSet((train_x, train_y), seed=seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=max(batch_size,128))


    train_loader_no_augment = DataLoader(train_data_no_augment, batch_size=batch_size, shuffle=True, drop_last=True)


    return train_loader, train_loader_no_augment, valid_loader, train_x, train_y, transform_train
