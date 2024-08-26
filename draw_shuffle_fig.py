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

parser.add_argument('--lr-decay', type=int, default=10000,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--save-folder', type=str, default=True,
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--exp-name', type=str, default="svhn",
                    help='exp-name (e.g "c_mnist", "m_mnist", "svhn", "cifar",'
                         '"fc100", "m_imgnet", "cub200")')

parser.add_argument('--target-label', type=int, default=3,
                    help='Which class to visualize.')
parser.add_argument('--data-num', type=int, default=10,
                    help='data number')
parser.add_argument('--vis-data-num', type=int, default=6,
                    help='how many data to visualize')

parser.add_argument('--predef_template', action="store_true",
                    default=False, help="use predefined template")
#seed 2, label 1  ==> CIFAR
#seed 6, label 8  ==> SVHN
#seed 0, label 1
#seed 4, label 4 ==> FC100

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

load_epoch_folder = '{}/exp_{}_prop_tr_gen_{}_{}/'.format('epoch_result',
                                                               args.exp_name,
                                                          args.data_num, args.seed);

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

def forward_shuffle(datas, labels, figures) :
    Critic_net.eval()
    Transformer_net.eval()
    Generator_net.eval();

    if args.cuda:
        data, label = datas.cuda(), labels.cuda()

    data, label = Variable(data), Variable(label)

    _ = Critic_net(data);

    total_transformed_list = list();
    total_spatial_transformed_list = list();

    templates = Generator_net(label);

    for src_i in range(len(datas)) :
        row_transformed_list = list();
        row_spatial_transformed_list = list();

        for tar_i in range(len(datas)) :
            src_dat = data[src_i];
            template = templates[src_i];
            tar_dat = data[tar_i];

            spatial_transformed, pixel_offset = Transformer_net.forward_spatial(template.unsqueeze(0), src_dat.unsqueeze(0),
                                                                                label[src_i].unsqueeze(0));

            spatial_transformed2, _ = Transformer_net.forward_spatial(template.unsqueeze(0), tar_dat.unsqueeze(0),
                                                                                label[tar_i].unsqueeze(0));


            final_transformed, cls_palette, palette_offset  = Transformer_net.forward_value(spatial_transformed,
                                                                                           tar_dat.unsqueeze(0),
                                                                                            spatial_transformed2,
                                                                                           label[src_i].unsqueeze(0));

            row_transformed_list.append(final_transformed[0]);
            row_spatial_transformed_list.append(spatial_transformed[0]);

        total_transformed_list.append(torch.stack(row_transformed_list));
        total_spatial_transformed_list.append(torch.stack(row_spatial_transformed_list));


    figure_shuffle = draw_shuffle_array(torch.clamp(torch.stack(total_transformed_list), min=0.0, max=1.0),
                       data, plt_save=True)
    figures.append(figure_shuffle)


def forward_interp(datas, labels, figures) :
    Critic_net.eval()
    Transformer_net.eval()
    Generator_net.eval();

    if args.cuda:
        data, label = datas.cuda(), labels.cuda()

    data, label = Variable(data), Variable(label)

    _ = Critic_net(data);


    tot_spat_interp_list = list();
    tot_appr_interp_list = list();

    templates = Generator_net(label);

    for tar_i in range(len(datas)):
        tar_dat = data[tar_i];

        spat_interp_list = list();

        appr_interp_list = list();
        #interp_vals = np.linspace(start=0.0, stop=1.0, num=4)
        interp_vals = np.array([0.0, 0.1, 0.3, 1.0])

        for iv in interp_vals:

            spatial_transformed, pixel_offset = Transformer_net.forward_spatial(templates[tar_i].unsqueeze(0),
                                                                                         tar_dat.unsqueeze(0),
                                                                                         label[tar_i].unsqueeze(0),
                                                                                         spatial_interp=iv);
            final_transformed, cls_palette, palette_offset = Transformer_net.forward_value(spatial_transformed,
                                                                                           tar_dat.unsqueeze(0),
                                                                                           spatial_transformed,
                                                                                           label[tar_i].unsqueeze(0));


            spat_interp_list.append(final_transformed[0]);

            spatial_transformed,  pixel_offset = Transformer_net.forward_spatial(templates[tar_i].unsqueeze(0),
                                                                                         tar_dat.unsqueeze(0),
                                                                                         label[tar_i].unsqueeze(0));
            final_transformed, cls_palette, palette_offset = Transformer_net.forward_value(spatial_transformed,
                                                                                           tar_dat.unsqueeze(0),
                                                                                           spatial_transformed,
                                                                                           label[tar_i].unsqueeze(0),
                                                                                           value_interp=iv);

            appr_interp_list.append(final_transformed[0]);

        tot_spat_interp_list.append(torch.stack(spat_interp_list));
        tot_appr_interp_list.append(torch.stack(appr_interp_list));


    fig_sp_interp = draw_interp_array(torch.clamp(torch.stack(tot_spat_interp_list), min=0.0, max=1.0),
                                        data, plt_save=False)
    figures.append(fig_sp_interp)

    fig_appr_interp = draw_interp_array(torch.clamp(torch.stack(tot_appr_interp_list), min=0.0, max=1.0),
                                      data, plt_save=True)
    figures.append(fig_appr_interp)

t_total = time.time()
best_train_loss = np.inf
best_epoch = 0

first_visit = False;
t = time.time()


figures = []
idxes = torch.where(train_y== args.target_label)[0]

with torch.no_grad() :

    forward_shuffle(train_x[idxes[:args.vis_data_num]], train_y[idxes[:args.vis_data_num]], figures);
    forward_interp(train_x[idxes[:args.vis_data_num]], train_y[idxes[:args.vis_data_num]], figures);


print("Visualization Finished!")




