from __future__ import division
from __future__ import print_function

import time
import argparse
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

import os
from modules.modules_GLTN import *
from modules_critic.conv4d import CNN_network_bn
from util.utils import draw_shuffle_array, draw_interp_array
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')   ##  59 good
parser.add_argument('--save-folder', type=str, default=True,
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--exp-name', type=str, default="svhn",
                    help='exp-name (e.g "c_mnist", "m_mnist", "svhn", "cifar",'
                         '"fc100", "m_imgnet", "cub200")')

parser.add_argument('--batch-size', type=int, default=32,
                    help='Number of samples per batch.')
parser.add_argument('--data-num', type=int, default=10,
                    help='data number')

parser.add_argument('--predef_template', action="store_true",
                    default=False, help="use predefined template")
#seed 2, label 1  ==> CIFAR
#seed 6, label 8  ==> SVHN
#seed 0, label 1
### gray or RGB
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


now = datetime.datetime.now()
timestamp = now.isoformat()

load_epoch_folder = '{}/exp_{}_prop_tr_gen_{}_{}/'.format('epoch_result', args.exp_name, args.data_num, args.seed);

transformer_file = os.path.join(load_epoch_folder, 'Transformer.pt')
generator_nn_file = os.path.join(load_epoch_folder, 'Generator.pt')
critic_nn_file = os.path.join(load_epoch_folder, "Critic.pt")


# Load Dataset
if args.exp_name =="c_mnist" : # color mnist dataset
    from util.data_loader import load_data_c_mnist

    n_z = 10;
    template_dim = 2
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_c_mnist(
        args.batch_size, train_size=6, val_size=20, seed=args.seed);

elif args.exp_name == "m_mnist" :  # moving mnist
    from util.data_loader import load_data_m_mnist

    n_z = 10;
    template_dim = 2
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_m_mnist(
        args.batch_size, train_size=6, val_size=20, seed=args.seed);

elif args.exp_name == "mnist" :  # mnist dataset
    from util.data_loader import load_data_mnist

    n_z = 10;
    template_dim=3
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_mnist(
        args.batch_size, train_size=args.data_num, val_size=1000, seed=args.seed);

elif args.exp_name == "fmnist" :  # fmnist dataset
    from util.data_loader import load_data_fmnist
    train_loader, valid_loader, train_x, train_y, transform_train = load_data_fmnist(
        args.batch_size, train_size= args.data_num, val_size=1000, seed=args.seed);

    n_z = 10;
    r_z = 0;
    template_dim=3

elif args.exp_name == "svhn": # svhn dataset
    from util.data_loader import load_data_svhn

    n_z = 10;
    template_dim=2
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_svhn(
        args.batch_size, train_size=args.data_num, val_size=200, seed=args.seed);
elif args.exp_name == "cifar":  # cifar dataset
    from util.data_loader import load_data_cifar

    n_z = 10;
    r_z = 5;
    template_dim=3

    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_cifar(
        args.batch_size, train_size=args.data_num, val_size=200, seed=args.seed);
elif args.exp_name == "fc100" :  # cifar100 dataset
    from util.data_loader import load_data_cifar100

    n_z = 5;
    r_z = 5;
    template_dim = 5

    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_cifar100(
        args.batch_size, train_size=args.data_num, val_size=100, seed=args.seed, num_classes=5);
elif args.exp_name == "cub200" :
    from util.data_loader import load_data_CUB200

    n_z = 5;
    r_z = 5;
    template_dim = 5

    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_CUB200(
        args.batch_size, train_size=args.data_num, val_size=10, seed=args.seed, num_classes=5);
# Load Model
if args.exp_name in ["c_mnist"] : # color mnist model
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=32, repeat_num=3, template_dim=template_dim);
    Transformer_net = Transformer(n_z, in_channel=3, channel_dim=32, repeat_num=2, template_dim= template_dim);
    Critic_net = CNN_network_bn(in_channel=3, channels=[8, 16, 32, 64], class_num = 10)
elif args.exp_name in ["m_mnist"] :
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=16, repeat_num=2, template_dim=template_dim);
    Transformer_net = Transformer(n_z, in_channel=1, channel_dim=16, repeat_num=2, template_dim= template_dim);
    Critic_net = CNN_network_bn(in_channel=1, channels=[8, 16, 32, 64], class_num = 10)

elif args.exp_name in ["mnist"] : # mnist model
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=64, repeat_num=2, template_dim=template_dim);
    Transformer_net = Transformer(n_z, in_channel=1, channel_dim=64, repeat_num=2, template_dim= template_dim);
    Critic_net = CNN_network_bn(in_channel=1, channels=[32, 64, 128, 256], class_num = 10)

elif args.exp_name in ["svhn"] :  # svhn model
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=64, repeat_num=3, template_dim=template_dim);
    Transformer_net = Transformer(n_z, in_channel=3, channel_dim=64, repeat_num=2, template_dim= template_dim);
    Critic_net = CNN_network_bn(in_channel=3, channels=[32, 64, 128, 256], class_num = n_z)

elif args.exp_name in ["cifar", "fc100"] : # cifar model
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=128, repeat_num=3, template_dim=template_dim);
    Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, repeat_num=2, template_dim= template_dim);
    Critic_net = CNN_network_bn(in_channel=3, channels=[32, 64, 128, 256], class_num =n_z)

elif args.exp_name in ["cub200"] :
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=256, repeat_num=3,
                              template_dim=template_dim, img_size=84, predef_template=args.predef_template);
    Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, img_size=84, repeat_num=2, repeat_num_v=2,
                                  template_dim=template_dim);
    Critic_net = CNN_network_bn(in_channel=3, channels=[32, 64, 128, 256], class_num=n_z)

Generator_net.load_state_dict(torch.load(generator_nn_file, map_location=torch.device('cpu')))
Transformer_net.load_state_dict(torch.load(transformer_file, map_location=torch.device('cpu')))
Critic_net.load_state_dict(torch.load(critic_nn_file, map_location=torch.device('cpu')))


criterion = CrossEntropyLoss()
criterion_D = nn.BCELoss()

if args.cuda:
    Transformer_net = Transformer_net.cuda()
    Generator_net = Generator_net.cuda()
    Critic_net = Critic_net.cuda()

def forward_shuffle(datas, labels) :
    Critic_net.eval()
    Transformer_net.eval()
    Generator_net.eval();

    if args.cuda:
        data, label = datas.cuda(), labels.cuda()

    data, label = Variable(data), Variable(label)

    _ = Critic_net(data);

    total_gens = list();

    templates = Generator_net(label);


    for src_i in range(len(datas)) :

        src_dat = data[src_i].unsqueeze(0).repeat(len(datas),1,1,1);

        spatial_transformed, pixel_offset = Transformer_net.forward_spatial(templates, src_dat,
                                                                            label);

        spatial_transformed2, _ = Transformer_net.forward_spatial(templates, data,
                                                                    label);


        final_transformed, cls_palette, palette_offset  = Transformer_net.forward_value(spatial_transformed,
                                                                                       data,
                                                                                        spatial_transformed2,
                                                                                       label);

        total_gens.append(final_transformed);


    return torch.cat(total_gens)


t_total = time.time()
best_train_loss = np.inf
best_epoch = 0

first_visit = False;
t = time.time()



with torch.no_grad() :
    ys = np.unique(train_y.cpu().data.numpy());

    for y_idx, y in enumerate(ys) :
        idxes = torch.where(train_y == y);

        gens = forward_shuffle(train_x[idxes], train_y[idxes]);


        np.save(os.path.join(load_epoch_folder, str(y_idx)), gens.cpu().data.numpy())



print("Visualization Finished!")




