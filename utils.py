
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch.nn.functional as F
import torch.utils.data as data
import random
from datas import *
import pickle

import seaborn as sns


from torchvision import transforms

import torchvision.datasets.mnist as Mnist
import torchvision.datasets.svhn as Svhn
import torchvision.datasets.cifar as cifar
import torchvision.datasets.stl10 as stl

from skimage import draw
import PIL.ImageDraw as ImageDraw
#import torchxrayvision as xray_vision

import torchvision.datasets.omniglot as omniglot


from PIL import Image, ImageFile
from scipy import ndimage

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)


class CustomDataSet(data.Dataset) :

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data[0];
        self.targets = data[1];


    def __getitem__(self, index):

        if isinstance(index, list) :
            idx = index[0];
            img, target = self.data[idx], self.targets[idx]
        else:
            img, target = self.data[index], self.targets[index]


        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class nonPermuteSampler(Sampler) :
    r""" non_permute the samples  and randomly iter.

        Arguments:
            data_source (Dataset): dataset to sample from
        """

    def __init__(self, data_source, n_sequence=5, batch_size=10):
        self.data_source = data_source
        self.n_sequence = n_sequence;
        self.save_idx = 0;

    def __iter__(self):
        n = len(self.data_source)

        sampler = iter(torch.arange(n)[self.save_idx:self.save_idx + self.n_sequence].tolist());
        self.save_idx += self.n_sequence;
        if self.save_idx >= n :
            self.save_idx = 0;

        return sampler

    def __len__(self):
        return len(self.data_source)

class PermuteSampler(Sampler):
    r""" permute the samples  and randomly iter.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source , n_sequence = 5, batch_size =10):
        self.data_source = data_source
        self.n_sequence = n_sequence;
        self.domain_idx = dict();

        idx_count = 0 ;
        for data in data_source :
            if data[1].item() in self.domain_idx :
                self.domain_idx[data[1].item()].append(idx_count);
            else :
                self.domain_idx[data[1].item()] = list();
                self.domain_idx[data[1].item()].append(idx_count);

            idx_count+=1;

        self.n_domain = len(self.domain_idx);

    def __iter__(self):

        # rand_class
        rand_class_idx = np.random.permutation(self.n_domain)[0];

        selected_domain_idx = self.domain_idx[rand_class_idx];

        return iter([selected_domain_idx[i] for i in np.random.permutation(len(selected_domain_idx))[:self.n_sequence]]);

    def __len__(self):
        return self.n_sequence

class BatchPermuteSampler(Sampler) :
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    '''
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    '''

    def __iter__(self):
        batch = []
        for b in range(self.batch_size) :
            b_list = [];
            for idx in self.sampler:
                b_list.append(idx)

            batch.append(b_list);

        yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def load_data_toy_mnist_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv', c_mode='move') :
    data = mnist();

    train_x, train_y, val_x, val_y, test_x, test_y, train_data_list, train_label_list, val_data_list, val_label_list = data.load_target_feature_data(
        train_size, val_size, test_size, target_class, mode= mode, seed=seed);

    src_dat = train_x[0];

    toy_train_list = list();

    toy_train_y_list = list();

    temp_source_val = np.linspace(0,5000, 784).astype(np.int32);
    temp_source_val = np.reshape(temp_source_val, [28, 28]);

    temp_list = list();

    if c_mode == 'move' :

        for i in range(train_size) :
            if mode == 'fc' :
                #src_dat_new = ndimage.rotate(np.squeeze(src_dat.reshape([28,28])), 50 * (i + 1), reshape=False)
                src_dat_new = ndimage.shift(np.squeeze(src_dat.reshape([28,28])), 2 * (i-2));
                toy_train_list.append(src_dat_new.reshape([784]));
                label_i = np.zeros(train_size, );
                label_i[i] = 1.0;
                toy_train_y_list.append(label_i)
            else :
                #src_dat_new = ndimage.rotate(np.squeeze(src_dat), 50*(i+1), reshape= False);
                src_dat_new = ndimage.shift(np.squeeze(src_dat), -2 * (i-2));

                temp_transform = ndimage.shift(np.squeeze(temp_source_val), 2 * (i));

                toy_train_list.append(np.expand_dims(src_dat_new,0));
                temp_list.append(np.expand_dims(temp_transform,0));


                label_i = np.zeros(train_size, );
                label_i[i] = 1.0;
                toy_train_y_list.append(label_i)

    elif c_mode == 'rot' :
        for i in range(train_size) :
            if mode == 'fc' :
                src_dat_new = ndimage.rotate(np.squeeze(src_dat.reshape([28,28])), 20 * (i-2), reshape=False)

                toy_train_list.append(src_dat_new.reshape([784]));
                label_i = np.zeros(train_size, );
                label_i[i] = 1.0;
                toy_train_y_list.append(label_i)
            else :
                src_dat_new = ndimage.rotate(np.squeeze(src_dat), 20*(i-2), reshape= False);

                temp_transform = ndimage.rotate(np.squeeze(temp_source_val), 20*(i), reshape= False);

                toy_train_list.append(np.expand_dims(src_dat_new, 0));
                temp_list.append(np.expand_dims(temp_transform, 0));

                label_i = np.zeros(train_size, );
                label_i[i] = 1.0;
                toy_train_y_list.append(label_i)


    train_x = np.stack(toy_train_list);
    train_y = np.stack(toy_train_y_list);
    train_data_list = np.stack(toy_train_list);


    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)

    label_train = torch.FloatTensor(np.argmax(train_y,1))
    label_val = torch.FloatTensor(np.argmax(val_y,1))


    train_data = TensorDataset(train_x, label_train)
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)


    return train_data_loader, valid_data_loader, train_data_list, train_label_list, val_data_list, val_label_list;


def load_data_mnist_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, seed = 25, mode= 'conv') :
    data = mnist();


    train_x_list = list();
    train_y_list = list();

    for c_idx in range(10) :

        train_x, train_y, val_x, val_y, test_x, test_y, train_data_list, train_label_list, val_data_list, val_label_list = data.load_target_feature_data(
            train_size, val_size, test_size, c_idx, mode=mode, seed=seed);

        train_x_list.extend(train_x);
        train_y_list.extend(np.argmax(train_y,1));

    train_x = np.stack(train_x_list);
    train_y = np.stack(train_y_list);

    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(train_x)

    label_train = torch.LongTensor(train_y)
    label_val = torch.LongTensor(train_y)

    train_data = TensorDataset(train_x, label_train)
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, valid_data_loader, train_x, train_y, val_x, val_y;


def load_data_svhn_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=(0.15)),
        transforms.ToTensor(),
    ])

    trainset = Svhn.SVHN(root='./dataset', download=True, split="train");
    valset = Svhn.SVHN(root='./dataset', download=True, split="test");

    train_x_list = list();
    train_y_list = list();

    for c_idx in range(10) :

        np.random.seed(seed)
        torch.manual_seed(seed)

        tot_train_data = trainset.data
        tot_train_label = trainset.labels

        train_label_idx = (tot_train_label == c_idx).nonzero()[0];
        random_idx = torch.randperm(train_label_idx.shape[0])
        train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

        train_x = tot_train_data[train_label_idx].astype(np.float) / 255.0;
        train_y = tot_train_label[train_label_idx];

        train_x_list.extend(train_x);
        train_y_list.extend(train_y);

    train_x = np.stack(train_x_list);
    train_y = np.stack(train_y_list)

    train_x = torch.FloatTensor(train_x);
    train_y = torch.LongTensor(train_y);

    val_x = train_x;
    val_y = train_y;

    disc_train_data = CustomDataSet((train_x, train_y), transform=transform_train)

    disc_data_loader = DataLoader(disc_train_data, batch_size=batch_size, shuffle=True)

    train_data = TensorDataset(train_x, train_y)
    valid_data = TensorDataset(val_x, val_y)
    '''
    if train_size > sequence_num:
        train_sampler = nonPermuteSampler(train_data, n_sequence=sequence_num);
        valid_sampler = nonPermuteSampler(valid_data, n_sequence=sequence_num);
    else:
        train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
        valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);
    '''

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, disc_data_loader, train_x, train_y, val_x, val_y;




def load_data_stl_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :
    trainset = stl.STL10(root='./dataset', download=True, split="train");
    valset = stl.STL10(root='./dataset', download=True, split="test");

    np.random.seed(seed)
    torch.manual_seed(seed)

    tot_train_data = trainset.data
    tot_train_label = trainset.labels

    train_label_idx = (tot_train_label == target_class).nonzero()[0];
    random_idx = torch.randperm(train_label_idx.shape[0])
    train_label_idx = train_label_idx[random_idx[:train_size]].squeeze();

    train_x = tot_train_data[train_label_idx].astype(np.float) / 255.0;
    train_y = tot_train_label[train_label_idx];

    tot_val_data = valset.data
    tot_val_label = valset.labels;

    val_label_idx = (tot_val_label == target_class).nonzero()[0];
    random_idx = torch.randperm(val_label_idx.shape[0])
    val_label_idx = val_label_idx[random_idx[:val_size]].squeeze();

    val_x = tot_val_data[val_label_idx].astype(np.float) / 255.0;
    val_y = tot_val_label[val_label_idx];

    train_data = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    valid_data = TensorDataset(torch.FloatTensor(val_x), torch.LongTensor(val_y))

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, valid_data_loader, train_x, train_y, val_x, val_y;



def load_data_fmnist_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, seed = 25, mode= 'conv') :

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


    trainset = Mnist.FashionMNIST(root='./dataset', download=True, train=True);
    valset = Mnist.FashionMNIST(root='./dataset', download=True, train=False);



    tot_train_data = trainset.train_data
    tot_train_label = trainset.targets

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

        train_x = tot_train_data[train_label_idx].unsqueeze(1).float()/255.0;
        train_y = tot_train_label[train_label_idx];

        train_x_list.extend(train_x);
        train_y_list.extend(train_y);


    train_x = torch.stack(train_x_list);
    train_y = torch.stack(train_y_list);

    val_x  = train_x;
    val_y  = train_y;

    disc_train_data = CustomDataSet((train_x, train_y), transform=transform_train)

    disc_data_loader = DataLoader(disc_train_data, batch_size=batch_size, shuffle=True)

    train_data = TensorDataset(train_x, train_y)
    valid_data = TensorDataset(val_x, val_y)

    '''
    if train_size > sequence_num :
        train_sampler = nonPermuteSampler(train_data, n_sequence=sequence_num);
        valid_sampler = nonPermuteSampler(valid_data, n_sequence=sequence_num);
    else :
        train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
        valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);
    '''
    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler= train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)


    return train_data_loader, disc_data_loader, train_x.cpu().numpy(), train_y.cpu().numpy(), val_x.cpu().numpy(), val_y.cpu().numpy();


def load_data_custom_sequence(batch_size, train_size = 8, sequence_num= 4, seed = 25, is_rgb=True) :
    dir_name = "./dataset/custom/";

    custom_files = os.listdir(dir_name);

    random.seed(seed);
    random.shuffle(custom_files);

    reshape_size_x = 64;
    reshape_size_y = 64;

    train_data_list = list();
    train_label_list = list();

    val_data_list = list();
    val_label_list = list();

    for file_idx in range(len(custom_files)):

        file_name = custom_files[file_idx];

        if is_rgb == True :
            image_np = np.array(Image.open(dir_name + file_name).convert('RGB').resize((reshape_size_x,reshape_size_y)))/255.0;
        else:

            image_np = np.array(Image.open(dir_name + file_name).resize((reshape_size_x, reshape_size_y))) / 255.0;

        if file_idx < train_size:
            train_data_list.append(image_np);
            train_label_list.append(1);  # dummy label

        if file_idx < train_size:  # we don't consider a validation data for this dataset
            val_data_list.append(image_np);
            val_label_list.append(1);  # dummy label

    if is_rgb :
        train_x = np.transpose(np.stack(train_data_list), [0, 3, 1, 2]);
        val_x = np.transpose(np.stack(val_data_list), [0, 3, 1, 2]);
    else :
        train_x = np.expand_dims(np.stack(train_data_list),1);
        val_x = np.expand_dims(np.stack(val_data_list),1);

    train_y = np.stack(train_label_list);
    val_y = np.stack(val_label_list);

    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)
    label_train = torch.LongTensor(train_y)
    label_val = torch.LongTensor(val_y)

    train_data = TensorDataset(train_x, label_train)  # label doesn't need figr is only for generation
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, valid_data_loader, np.transpose(np.stack(train_data_list), [0, 3, 1,
                                                                                          2]), train_label_list, val_data_list, val_label_list;


def load_data_mini_img_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :

    np.random.seed(seed);

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
    ])

    dataset_path = "./datset/mini_imgnet_/";

    train_in = open(dataset_path +"mini-imagenet-cache-train.pkl", "rb")
    val_in = open(dataset_path + "mini-imagenet-cache-val.pkl", "rb")
    label_string_file = open(dataset_path + "map_clsloc.txt")

    id_to_label = dict();

    for line_num in range(1000) :
        line = label_string_file.readline();

        splitted_info = line.split(" "); # split with space
        id_to_label[splitted_info[0]] = splitted_info[2];

    train_pickle = pickle.load(train_in);

    train_img = train_pickle['image_data'];
    train_label_dict = train_pickle['class_dict'];

    total_class_num = len(train_label_dict);

    tot_class_list = np.arange(total_class_num);
    np.random.shuffle(tot_class_list);

    target_class_idx = tot_class_list[:5];
    class_key_list = list(train_pickle['class_dict'].keys());

    train_data_list = list();
    train_label_list = list();


    for c_count, c_idx in enumerate(target_class_idx) :
        np.random.seed(seed);

        class_nid = class_key_list[c_idx];
        class_name = id_to_label[class_nid];
        print(class_name);

        data_idx = np.stack(train_label_dict[class_nid]);

        tot_class_idx = np.arange(len(data_idx));
        np.random.shuffle(tot_class_idx);
        train_dat_idx = data_idx[tot_class_idx[:train_size]];

        train_img_data = train_img[train_dat_idx];

        train_data_list.append(np.transpose(train_img_data/255.0, (0,3,1,2)));
        train_label_list.extend(np.ones(train_size, dtype=np.int32) * c_count);

    train_dat = np.vstack(train_data_list);
    train_label = np.stack(train_label_list);
    val_dat = train_dat;
    val_label = train_label;

    train_x = torch.FloatTensor(train_dat);
    val_x = torch.FloatTensor(val_dat);
    label_train = torch.LongTensor(train_label);
    label_valid = torch.LongTensor(val_label);

    #train_data = TensorDataset(train_x, label_train)
    train_data = TensorDataset(train_x, label_train);
    valid_data = TensorDataset(val_x, label_valid)

    disc_train_data = CustomDataSet((train_x, label_train), transform=None)

    disc_data_loader = DataLoader(disc_train_data, batch_size=batch_size, shuffle=True)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, disc_data_loader, train_dat, train_label, val_dat, val_label;



def load_data_cifar_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trainset = cifar.CIFAR10(root='./datset', train=True, download=True)
    valset = cifar.CIFAR10(root='./datset', train=False, download=True)

    train_img = trainset.data;
    train_label = np.stack(trainset.targets);

    val_img = valset.data
    val_label = np.stack(valset.targets);

    train_data_list = list();
    train_label_list = list();

    val_data_list = list();
    val_label_list = list();
    c_count = 0;
    for c in range(10):
        np.random.seed(seed);
        idx_c = np.where(train_label == c)[0];
        # np.random.shuffle(idx_c);
        idx_c = np.random.permutation(idx_c);
        img_list = list();
        for ii in range(train_size):
            resized_img = np.transpose((train_img[idx_c[ii]]) / 255.0, (2, 0, 1));
            img_list.append(resized_img);

        train_data_list.append(img_list);
        # np.tile(np.ones(len(train_img_list[c_count]), dtype=np.int32) * c_count
        train_label_list.extend(np.tile(np.ones(train_size, dtype=np.int32) * c_count, 1));

        idx_c = np.where(val_label == c)[0];

        val_data_list.append(np.transpose((val_img[idx_c]) / 255.0, (0, 3, 1, 2)));
        val_label_list.extend(np.ones(len(val_data_list[c_count]), dtype=np.int32) * c_count)

        c_count += 1;

    train_dat = np.vstack(train_data_list);
    train_label = np.stack(train_label_list);
    val_dat = train_dat;
    val_label = train_label;

    train_x = torch.FloatTensor(train_dat);
    val_x = torch.FloatTensor(val_dat);
    label_train = torch.LongTensor(train_label);
    label_valid = torch.LongTensor(val_label);

    #train_data = TensorDataset(train_x, label_train)
    train_data = TensorDataset(train_x, label_train);
    valid_data = TensorDataset(val_x, label_valid)

    disc_train_data = CustomDataSet((train_x, label_train), transform=transform_train)

    disc_data_loader = DataLoader(disc_train_data, batch_size=batch_size, shuffle=True)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, disc_data_loader, train_dat, train_label, val_dat, val_label;


def covid_img_unnormalize(img, maxval) :

    sample = ((img/1024 + 1.)/2.) * maxval


    return sample

def load_data_covid19_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num=4, target_class=1, seed=25, mode='conv') :

    data = COVID19_Dataset(imgpath="./dataset/covid_data/images",
                                                csvpath="./dataset/covid_data/metadata.csv",
                                                views=["PA", "AP"],

                                                transform=None,
                                                data_aug=None,
                                                nrows=None,
                                                seed=seed,
                                                pure_labels=False,
                                                unique_patients=True
                                                )

    pathodology_to_label_idx = dict();

    pathodology_to_label_idx['COVID-19'] = 2;
    pathodology_to_label_idx['SARS'] = 15;
    pathodology_to_label_idx['MERS'] = 10;
    pathodology_to_label_idx["Influenza"] = 6;
    pathodology_to_label_idx["Varicella"] = 17;
    pathodology_to_label_idx["No Finding"] = 12;
    pathodology_to_label_idx["Viral"] = 18;
    pathodology_to_label_idx['Bacterial'] = 1;
    pathodology_to_label_idx['Fungal'] = 5;
    pathodology_to_label_idx['Lipoid'] = 9;

    label_list = list();
    img_list = list();

    for dat in data :

        flag_COVID = dat['lab'][pathodology_to_label_idx['COVID-19']]
        flag_SARS = dat['lab'][pathodology_to_label_idx['SARS']]
        flag_MERS = dat['lab'][pathodology_to_label_idx['MERS']]
        flag_Influenza = dat['lab'][pathodology_to_label_idx['Influenza']]
        flag_Varicella = dat['lab'][pathodology_to_label_idx['Varicella']]

        flag_Bacterial = dat['lab'][pathodology_to_label_idx['Bacterial']]
        flag_Fungal = dat['lab'][pathodology_to_label_idx['Fungal']]
        flag_Lipoid = dat['lab'][pathodology_to_label_idx['Lipoid']]

        flag_healty = dat['lab'][pathodology_to_label_idx['No Finding']]

        flags = np.asarray([flag_COVID, flag_SARS, flag_MERS, flag_Influenza, flag_Varicella, flag_Bacterial, flag_Fungal, flag_Lipoid, flag_healty]);

        img = covid_img_unnormalize(dat['img'], 255);

        img = Image.fromarray(np.squeeze(img).astype('uint8'), 'L');
        img = img.resize((224,224));


        label_list.append(flags);
        img_list.append(np.expand_dims(np.array(img)/255.0, 0));

    label_arry = np.stack(label_list);
    img_arry = np.expand_dims(np.vstack(img_list),1);

    tot_dat_num = label_arry.shape[0];

    total_idx = np.arange(tot_dat_num);

    np.random.seed(seed);
    np.random.shuffle(total_idx);

    img_arry = img_arry[total_idx];
    label_arry = np.argmax(label_arry[total_idx],1);

    for target_cl in range(1,9) :

        #target_idx = np.where(label_arry == target_class)[0];
        target_idx = np.where(label_arry == target_cl)[0];
        min_train_num = 3;

        target_num = target_idx.shape[0]

        if int(target_num*0.8) > min_train_num :
            train_idx = target_idx[:int(target_num*0.8)];
            val_idx = target_idx[int(target_num*0.8):];
        else :
            train_idx = target_idx[:3];
            val_idx = target_idx[3:];

        train_x = img_arry[train_idx];
        train_y = label_arry[train_idx];

    val_x = img_arry[val_idx];
    val_y = label_arry[val_idx];

    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)
    label_train = torch.LongTensor(train_y)
    label_val = torch.LongTensor(val_y)


    train_data = TensorDataset(train_x, label_train)  # label doesn't need figr is only for generation
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, valid_data_loader, img_arry[train_idx], label_arry[train_idx], img_arry[val_idx], label_arry[val_idx];


    train_data_list = list();

    train_data_list.append(img_arry[target_idx]);
    train_label_list.append(1);  # dummy label









def load_data_omniglot_sequence(batch_size, train_size = 8, val_size = 10, test_size=10, sequence_num= 4, target_class=1, seed = 25, mode= 'conv') :
    data = Omniglot(root='./dataset', download=True, background=False, transform=transforms.Compose([lambda x: x.resize((28, 28))]))

    tot_img_list = list();
    tot_label_list = list();

    for (img, label) in data :
        tot_img_list.append(img);
        tot_label_list.append(label);

    tot_img = np.stack(tot_img_list);
    tot_label = np.stack(tot_label_list);

    train_data_list = list();
    train_label_list = list();
    val_data_list = list();
    val_label_list = list();

    for c in target_class:

        np.random.seed(seed);
        idx_c = np.where(tot_label == c)[0];
        # np.random.shuffle(idx_c);
        idx_c = np.random.permutation(idx_c);
        img_list = list();
        img_val_list = list();
        for ii in range(train_size) :
            resized_img = 1 - np.squeeze(tot_img[idx_c[ii]])/255.0;
            img_list.append(np.expand_dims(resized_img,0));

        for ii in range(train_size, idx_c.shape[0]) :
            resized_img = 1 - np.squeeze(tot_img[idx_c[ii]])/255.0;
            img_val_list.append(np.expand_dims(resized_img,0));


        train_data_list.extend(img_list);
        train_label_list.extend(tot_label[idx_c[:train_size]]);

        val_data_list.extend(img_val_list);
        val_label_list.extend(tot_label[idx_c[train_size:]]);

    train_dat = np.stack(train_data_list);
    train_label = np.stack(train_label_list);

    val_dat = np.stack(val_data_list);
    val_label = np.stack(val_label_list);


    train_x = train_dat;
    val_x = val_dat;


    train_x = torch.FloatTensor(train_x)
    val_x = torch.FloatTensor(val_x)

    label_train = torch.FloatTensor(train_label)
    label_val = torch.FloatTensor(val_label)


    train_data = TensorDataset(train_x, label_train)
    valid_data = TensorDataset(val_x, label_val)

    train_sampler = PermuteSampler(train_data, n_sequence=sequence_num);
    valid_sampler = PermuteSampler(valid_data, n_sequence=sequence_num);

    train_batch_sampler = BatchPermuteSampler(train_sampler, batch_size=batch_size, drop_last=False);
    valid_batch_sampler = BatchPermuteSampler(valid_sampler, batch_size=batch_size, drop_last=False);

    train_data_loader = DataLoader(train_data, batch_sampler=train_batch_sampler)
    valid_data_loader = DataLoader(valid_data, batch_sampler=valid_batch_sampler)

    return train_data_loader, valid_data_loader, train_dat, train_label_list, val_data_list, val_label_list;

def rand_insert(data, template) :
    insert_data_list = list();
    rand_idx_list = list();

    for batch_idx in range(data.shape[0]) :
        rand_idx = np.random.randint(1, data.shape[1]+1);

        insert_data_list.append(torch.cat([data[batch_idx, :rand_idx], template[batch_idx], data[batch_idx, rand_idx:]], dim=0));
        rand_idx_list.append(rand_idx)

    rand_idx = np.stack(rand_idx_list);

    torch_idx = torch.from_numpy(rand_idx).long();

    if data.is_cuda :
        torch_idx = torch_idx.cuda();

    return torch.stack(insert_data_list), torch_idx;

def interpolate_vector(tar_transform_param, gen_num=10) :

    src_zero_vector = torch.zeros_like(tar_transform_param);

    if tar_transform_param.is_cuda :
        src_zero_vector = src_zero_vector.cuda();

    unit_interpol = tar_transform_param - src_zero_vector;

    interpol_rate = np.linspace(0.0, 1.0, gen_num);

    interpol_vectors = list();
    for i in range(gen_num) :
        interpol_vectors.append(unit_interpol * interpol_rate[i]);

    return torch.stack(interpol_vectors,0)


def draw_templates(templates, c_idx) :

    fig, axes = plt.subplots(10, 10, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    for i in range(10):

        for j in range(10):

            if templates.shape[-3] == 3:

                axes[i, j].imshow(np.transpose(templates[i*10+j].data.cpu().numpy(), (1, 2, 0)))
            else:
                axes[i, j].imshow(np.squeeze(templates[i*10+j].data.cpu().numpy()), cmap='Greys')
            axes[i, j].axis('off')

    plt.savefig("fig/template" + str(c_idx) + ".jpg");
    plt.close('all')

def draw_shuffle_temp( transformed, target, c_idx) :
    target = target.squeeze(0)
    fig, axes = plt.subplots(transformed.shape[1], transformed.shape[0] + 4, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    for i in range(transformed.shape[1]):

        for j in range(transformed.shape[0] + 2):

            if j == transformed.shape[0]:  # draw target1
                draw_target = target[0];
            elif j == transformed.shape[0] + 1:  # draw target2
                draw_target = target[1];
            else:
                draw_target = transformed[j, i];

            if draw_target.shape[0] == 3:
                axes[j].imshow(np.transpose(draw_target.data.cpu().numpy(), (1, 2, 0)))
            else:
                axes[j].imshow(np.squeeze(draw_target.data.cpu().numpy()), cmap='Greys')
            axes[j].axis('off')

    plt.savefig("fig/result" + str(c_idx) + ".jpg");
    plt.close('all')


def draw_transformation_sequence(template, spatial_transformed, final_transformed, datas, c_idx, template_dim) :

    final_transformed = final_transformed.squeeze(2)
    spatial_transformed = spatial_transformed.squeeze(2)

    colors = sns.color_palette("colorblind", template_dim);

    np_template = template.cpu().data.numpy();
    np_spatial_transformed = spatial_transformed.cpu().data.numpy()

    color_mapped_template = np.zeros(shape=(template.shape[0], template.shape[1], 3));
    color_mapped_spatial_template = np.zeros(shape=(spatial_transformed.shape[0], spatial_transformed.shape[1],
                                                    template.shape[0], template.shape[1], 3));
    for idx, color in enumerate(colors):
        row, col = np.where(np_template == idx);
        color_mapped_template[row, col, :] = color;

        idx0, idx1, row, col = np.where(np_spatial_transformed == idx);
        color_mapped_spatial_template[idx0, idx1, row, col, :] = color;


    for i in range(final_transformed.shape[0]) :
        for j in range(final_transformed.shape[0]) :
            # src
            src_dat = color_mapped_template
            # tar
            tar_dat_1 = np.transpose(datas[i].cpu().data.numpy(), (1,2,0));
            spat_tr_1 = color_mapped_spatial_template[i,i];
            final_tr_1 = np.transpose(final_transformed[i, i].cpu().data.numpy(), (1,2,0));

            tar_dat_2 = np.transpose(datas[j].cpu().data.numpy(), (1,2,0));
            spat_tr_2 = color_mapped_spatial_template[j, j];
            final_tr_2 = np.transpose(final_transformed[j, j].cpu().data.numpy(), (1,2,0));

            for k in range(final_transformed.shape[0]) :

                fig, axes = plt.subplots(5, 7, gridspec_kw={'wspace': 0, 'hspace': 0},
                                         squeeze=True);
                fig.set_size_inches(12, 12)

                for pp in range(5) :
                    for qq in range(7) :
                        axes[pp,qq].axis('off');

                spat_tr_3 = color_mapped_spatial_template[i,j];
                final_tr_3 = np.transpose(final_transformed[i,j].cpu().data.numpy(), (1,2,0));

                axes[0,0].imshow(src_dat);
                axes[0,2].imshow(spat_tr_1);
                axes[0,4].imshow(final_tr_1);
                axes[0,6].imshow(tar_dat_1);

                axes[2,0].imshow(src_dat);
                axes[2,2].imshow(spat_tr_2);
                axes[2,4].imshow(final_tr_2);
                axes[2,6].imshow(tar_dat_2);

                axes[4,0].imshow(src_dat);
                axes[4,2].imshow(spat_tr_3);
                axes[4,4].imshow(final_tr_3);

                plt.savefig("fig/result" + str(c_idx) + "_" + str(i) +  "_" + str(j) + "_" + str(k)+ ".jpg");
                plt.close('all')


def draw_shuffle_array(template, transformed, datas, c_idx, template_dim) :

    transformed = transformed.squeeze(2)
    fig, axes = plt.subplots(transformed.shape[0]+1, transformed.shape[0]+1);
    fig.set_size_inches(6.5, 6)
    #fig.set_size_inches(13, 12)

    colors = sns.color_palette("colorblind", template_dim);

    np_template = template.cpu().data.numpy();
    #np_spatial_transformed = spatial_transformed.cpu().data.numpy()

    color_mapped_template = np.zeros(shape=(template.shape[0], template.shape[1], 3));
    #color_mapped_spatial_template = np.zeros(shape=(spatial_transformed.shape[0], spatial_transformed.shape[1],
    #                                                template.shape[0], template.shape[1], 3));
    for idx, color in enumerate(colors):
        row, col = np.where(np_template == idx);
        color_mapped_template[row, col, :] = color;

        #idx0, idx1, row, col = np.where(np_spatial_transformed == idx);
        #color_mapped_spatial_template[idx0, idx1, row, col, :] = color;


    axes[0,0].axis('off');
    draw_target = color_mapped_template;
    axes[0,0].imshow(draw_target);


    for i in range(transformed.shape[0]) : # fill value
        if datas.shape[1] == 3:
            axes[0,i+1].imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            axes[0,i+1].imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        axes[0,i+1].axis('off')

    for i in range(transformed.shape[0]) : # fill spatial
        if datas.shape[1] == 3:
            axes[i+1,0].imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            axes[i+1,0].imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        axes[i+1, 0].axis('off')


    for i in range(transformed.shape[0]) : # fill spatial
        pass
    for i in range(transformed.shape[0]) :
        for j in range(transformed.shape[0]) :
            if datas.shape[1] == 3:
                axes[i+1,j+1].imshow(np.transpose(transformed[i,j].data.cpu().numpy(), (1,2,0)))
            else :
                axes[i+1,j+1].imshow(np.squeeze(transformed[i,j].data.cpu().numpy()), cmap='Greys')

            axes[i+1, j+1].axis('off')

    for i in range(transformed.shape[0]+1) :
        for j in range(transformed.shape[0]+1) :
            axes[i,j].axis('off')
    plt.savefig("fig/" +str(c_idx) + "_result.jpg");
    plt.close('all')


def draw_spatial_segment(template, template_dim, spatial_transformed, datas, c_idx) :

    fig, axes = plt.subplots(2, spatial_transformed.shape[0] + 1, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    colors = sns.color_palette("dark", template_dim);

    np_template = template.cpu().data.numpy();
    np_spatial_transformed = spatial_transformed.cpu().data.numpy()

    color_mapped_template = np.zeros(shape=(template.shape[0], template.shape[1], 3));
    color_mapped_spatial_template = np.zeros(shape=(spatial_transformed.shape[0], spatial_transformed.shape[1],
                                                    template.shape[0], template.shape[1], 3));
    for idx, color in enumerate(colors):
        row, col = np.where(np_template == idx);
        color_mapped_template[row, col, :] = color;

        idx0, idx1, row, col = np.where(np_spatial_transformed == idx);
        color_mapped_spatial_template[idx0, idx1, row, col, :] = color;

    axes[0, 0].axis('off');
    #axes[0, 0].imshow(color_mapped_template);
    axes[1, 0].axis('off');
    #axes[1, 0].imshow(color_mapped_template);

    for i in range(spatial_transformed.shape[0]):  # fill value
        if datas.shape[1] == 3:
            axes[0, i + 1].imshow(np.transpose(datas[i].data.cpu().numpy(), (1, 2, 0)));
        else:
            axes[0, i + 1].imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        axes[0, i + 1].axis('off')


    for i in range(1):
        for j in range(5):
            axes[i + 1, j + 1].imshow(color_mapped_spatial_template[j, i])
            axes[i + 1, j + 1].axis('off')

    plt.savefig("fig/result" + str(c_idx) + ".jpg");
    plt.close('all')


def draw_segment_shuffle_array(template, template_dim, spatial_transformed, final_transformed, datas, c_idx) :

    transformed = final_transformed.squeeze(2)

    fig, axes = plt.subplots(transformed.shape[0]+1, transformed.shape[0]+1, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    colors = sns.color_palette("dark", template_dim);

    np_template = template.cpu().data.numpy();
    np_spatial_transformed = spatial_transformed.cpu().data.numpy()

    color_mapped_template = np.zeros(shape=(template.shape[0],template.shape[1],3));
    color_mapped_spatial_template = np.zeros(shape=(spatial_transformed.shape[0], spatial_transformed.shape[1],
                                                    template.shape[0],template.shape[1],3));
    for idx, color in enumerate(colors) :
        row,col = np.where(np_template == idx);
        color_mapped_template[row,col,:] = color;

        idx0, idx1, row,col = np.where(np_spatial_transformed==idx);
        color_mapped_spatial_template[idx0, idx1, row,col,:] = color;


    axes[0,0].axis('off');
    axes[0,0].imshow(color_mapped_template);


    for i in range(transformed.shape[0]) : # fill value
        if datas.shape[1] == 3:
            axes[0,i+1].imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            axes[0,i+1].imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        axes[0,i+1].axis('off')

    for i in range(transformed.shape[0]) : # fill spatial
        if datas.shape[1] == 3:
            axes[i+1,0].imshow(np.transpose(datas[i].data.cpu().numpy(), (1,2,0)));
        else :
            axes[i+1,0].imshow(np.squeeze(datas[i].data.cpu().numpy()), cmap='Greys')
        axes[i+1, 0].axis('off')


    for i in range(transformed.shape[0]) : # fill spatial
        pass
    for i in range(5) :
        for j in range(5) :
            if datas.shape[1] == 3:
                #axes[i+1,j+1].imshow(np.transpose(transformed[i,j].data.cpu().numpy(), (1,2,0)))
                axes[i+1,j+1].imshow(color_mapped_spatial_template[i,j])
            else :
                axes[i+1,j+1].imshow(np.squeeze(transformed[i,j].data.cpu().numpy()), cmap='Greys')

            axes[i+1, j+1].axis('off')

    plt.savefig("fig/" + str(c_idx) +"_result_segment.jpg");
    plt.close('all')



def draw_shuffle_list(template, spatial_transformed,  transformed, target, c_idx) :

    template = template.squeeze(0);                    # template
    target = target.squeeze(0)
    fig, axes = plt.subplots(transformed.shape[1], transformed.shape[0] +4, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    for i in range(transformed.shape[1]) :

        for j in range(transformed.shape[0] +4) :

            if j == 0 : # draw src
                draw_target = template;
            elif j==1 :
                draw_target = spatial_transformed
            elif j== transformed.shape[0] +2 : # draw target1
                draw_target = target[0];
            elif j== transformed.shape[0] +3 : # draw target2
                draw_target = target[1];
            else :
                draw_target = transformed[j-2,i];

            if draw_target.shape[0] == 3 :
                axes[j].imshow(np.transpose(draw_target.data.cpu().numpy(), (1,2,0)))
            else :
                axes[j].imshow(np.squeeze(draw_target.data.cpu().numpy()), cmap='Greys')
            axes[j].axis('off')

    plt.savefig("fig/result" + str(c_idx) + ".jpg");
    plt.close('all')


def draw_shuffle_list_onehot(template, spatial_template, default_colored, transformed, target, c_idx) :

    template = template.squeeze(0);                    # template
    target = target.squeeze(0)
    fig, axes = plt.subplots(transformed.shape[1], transformed.shape[0] +5, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    for i in range(transformed.shape[1]) :

        for j in range(transformed.shape[0] +5) :

            if j == 0 : # draw src
                draw_target = template;

            elif j == 1:  # draw src
                draw_target = spatial_template;

            elif j==2 : # draw default colored
                draw_target = default_colored[j-2,i];

            elif j== transformed.shape[0] +3 : # draw target1
                draw_target = target[0];
            elif j== transformed.shape[0] +4 : # draw target2
                draw_target = target[1];
            else :
                draw_target = transformed[j-3,i];

            if draw_target.shape[0] == 3 :
                axes[j].imshow(np.transpose(draw_target.data.cpu().numpy(), (1,2,0)))
            else :
                axes[j].imshow(np.squeeze(draw_target.data.cpu().numpy()), cmap='Greys')
            axes[j].axis('off')

    plt.savefig("fig/result" + str(c_idx) + ".jpg");
    plt.close('all')


def draw_interp_list(template, transformed, target, c_idx) :
    template = template.squeeze(0);  # template
    target = target.squeeze(0)
    fig, axes = plt.subplots(transformed.shape[1], transformed.shape[0] + 2, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    for i in range(transformed.shape[1]):

        for j in range(transformed.shape[0] + 2):

            if j == 0:  # draw src
                draw_target = template;
            elif j == transformed.shape[0] + 1:  # draw target1
                draw_target = target
            else:
                draw_target = transformed[j - 1, i];

            if draw_target.shape[0] == 3:
                axes[j].imshow(np.transpose(draw_target.data.cpu().numpy(), (1, 2, 0)))
            else:
                axes[j].imshow(np.squeeze(draw_target.data.cpu().numpy()), cmap='Greys')
            axes[j].axis('off')

    plt.savefig("fig/interp_result" + str(c_idx) + ".jpg");
    plt.close('all')

def draw_interp_all(template, transformed, target, c_idx) :

    template = template.squeeze(0);                    # template
    target = target.squeeze(0)
    fig, axes = plt.subplots(1, transformed.shape[0] +2, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    for i in range(1,2) :

        for j in range(transformed.shape[0] +2) :

            if j == 0 : # draw src
                draw_target = template;
            elif j== transformed.shape[0] +1 : # draw target
                draw_target = target[1];
            else :
                draw_target = transformed[j-1,0];

            if draw_target.shape[0] == 3 :
                axes[j].imshow(np.transpose(draw_target.data.cpu().numpy(), (1,2,0)))
            else :
                axes[j].imshow(np.squeeze(draw_target.data.cpu().numpy()), cmap='Greys')
            axes[j].axis('off')

    plt.savefig("fig/result" + str(c_idx) + ".jpg");
    plt.close('all')


def draw_gen_template(template) :
    fig, axes = plt.subplots(10, 10, gridspec_kw={'wspace': 0, 'hspace': 0},
                             squeeze=True);

    fig_count = 0;

    for i in range(10*10) :
        if template[i].shape[0] == 3 :
            axes[fig_count//10, fig_count%10].imshow(np.transpose(template[i].data.cpu().numpy(), (1,2,0)));
        else :
            axes[fig_count//10, fig_count%10].imshow(np.squeeze(template[i].data.cpu().numpy()), cmap='Greys');
        fig_count+=1;

    plt.savefig("fig/gen_template.jpg");




def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}

    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot;


def encode_onehot_rec(labels, classes):


    max_label = classes;
    n_label = list();

    label_idx = 0;

    for i in range(classes) :


        for j in range(i) :
            zero_arr = np.zeros(classes);

            if label_idx < labels[1].shape[0] :
                if labels[0][label_idx] == i and labels[1][label_idx] == j :
                    zero_arr[i] = 1.0;
                    label_idx +=1;

            n_label.append(zero_arr);

        for j in range(i+1, max_label) :
            zero_arr = np.zeros(classes);

            if label_idx < labels[1].shape[0]:
                if labels[0][label_idx] == i and labels[1][label_idx] == j and label_idx < labels[1].shape[0]:
                    zero_arr[i] = 1.0;
                    label_idx += 1;
            n_label.append(zero_arr)

    return np.stack(n_label)


def encode_onehot_sen(labels, classes):


    max_label = classes;
    n_label = list();

    label_idx = 0;

    for i in range(classes) :


        for j in range(i) :
            zero_arr = np.zeros(classes);

            if label_idx < labels[1].shape[0] :
                if labels[0][label_idx] == i and labels[1][label_idx] == j :
                    zero_arr[j] = 1.0;
                    label_idx +=1;

            n_label.append(zero_arr);

        for j in range(i+1, max_label) :
            zero_arr = np.zeros(classes);

            if label_idx < labels[1].shape[0]:
                if labels[0][label_idx] == i and labels[1][label_idx] == j and label_idx < labels[1].shape[0]:
                    zero_arr[j] = 1.0;
                    label_idx += 1;
            n_label.append(zero_arr)

    return np.stack(n_label)

def gradient_penalty(y, x) :
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    if y.is_cuda :
        weight = torch.ones(y.size()).cuda();
    else :
        weight= torch.ones(y.size())

    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)

def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()

def transform_alpha(z_alpha, cuda=True) :

    random_ratio = np.random.uniform(0, 1, size=(z_alpha.shape[0], z_alpha.shape[1]));
    random_ratio = torch.FloatTensor(torch.from_numpy(random_ratio.astype(np.float32)));

    if cuda :
        random_ratio =  random_ratio.cuda();

    return torch.sum(random_ratio.unsqueeze(-1) * z_alpha, 1);


def overlay_polygon(np_img, poly_type = "tri") :

    img_shape= np_img.shape;

    if poly_type == "tri" :
        while True :
            tri_x, tri_y = np.random.uniform(low=0.0, high=img_shape[-1], size=(2, 3))  # triangle
            rr, cc = draw.polygon(tri_x, tri_y, shape=img_shape[-2:]);
            if rr.shape[0] > int(np_img.shape[-1]*np_img.shape[-1]/10) and \
                rr.shape[0] < int(np_img.shape[-1] * np_img.shape[-1] / 2) :
                break;

    elif poly_type == "rect" :
        rect_cent_x, rect_cent_y  = np.random.uniform(low=0.0, high=img_shape[-1], size=(2, 1));  # rect center
        rect_len_x, rect_len_y = np.random.uniform(low=np_img.shape[-1]*0.1, high=np_img.shape[-1]*0.5, size=(2, 1));  # rect len
        rect_cent_x, rect_cent_y, rect_len_x, rect_len_y = int(rect_cent_x) , int(rect_cent_y), int(rect_len_x), int(rect_len_y);

        rr, cc = draw.rectangle(start=(rect_cent_x - int(rect_len_x*0.5),rect_cent_y - int(rect_len_y*0.5)), # center
                       extent=(rect_cent_x + int(0.5*rect_len_x), rect_cent_y + int(0.5*rect_len_y)), shape=img_shape[-2:]);

    elif poly_type == "elip" :
        elip_cent_x, elip_cent_y  = np.random.uniform(low=0.0, high=img_shape[-1], size=(2, 1));  # rect center
        elip_len_x, elip_len_y = np.random.uniform(low=np_img.shape[-1]*0.1, high=np_img.shape[-1]*0.5, size=(2, 1));  # rect len

        elip_cent_x, elip_cent_y, elip_len_x, elip_len_y = int(elip_cent_x) , int(elip_cent_y), \
                                                           int(elip_len_x*0.5), int(elip_len_y*0.5);
        rr, cc = draw.ellipse(elip_cent_x,elip_cent_y, # center
                       elip_len_x, elip_len_y, shape=img_shape[-2:]);


    np_img[rr,cc] = 1.0;

    return np_img;





def generate_polygons(img_shape, num_repeat) :

    total_pallete_list = list();

    for i in range(img_shape[0]) :
        pallete_list = list();
        for k in range(num_repeat) :
            pallete = np.zeros(shape=img_shape[-2:], dtype=np.float32);
            overlay_polygon(pallete, poly_type="rect")

            pallete_list.append(pallete);
        total_pallete_list.append(np.stack(pallete_list));


    return pallete







def generate_templates(r_z, batch_size, Template_net, cuda=True) :

    z_src = torch.rand(size=(batch_size, r_z));

    if cuda :
        z_src = z_src.cuda()

    gen_template = Template_net(z_src.unsqueeze(-1).unsqueeze(-1));

    #input_data = torch.cat([gen_template.unsqueeze(1), input_data], axis=1);

    return gen_template




