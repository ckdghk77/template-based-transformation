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
from util.utils import nll_gaussian, cosine_dist, cosine_sim
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')   ##  59 good
parser.add_argument('--epochs_gen', type=int, default=10000,
                    help='Number of epochs to train generator')
parser.add_argument('--epochs_clf', type=int, default=1000,
                    help='Number of epochs to train classifier')

parser.add_argument('--lr-decay', type=int, default=10000,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Initial learning rate.')
parser.add_argument('--save-folder', type=str, default=True,
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--exp-name', type=str, default="svhn",
                    help='exp-name (e.g "c_mnist", "m_mnist", "svhn", "cifar",'
                         '"fc100", "m_imgnet", "cub200")')

parser.add_argument('--data-num', type=int, default=10,
                    help='data number')

parser.add_argument('--vis-interval', type=int, default=50,
                    help="visualisation interval (for img)")
parser.add_argument('--save-interval', type=int, default=100,
                    help="save interval (for model)")

parser.add_argument('--tb-logging', action="store_true",
                    default=False, help="tensorboard logging flag")



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

run_name = 'exp_{}_prop_tr_{}_{}_{}/'.format(args.exp_name,
                                             "gen",
                                             args.data_num, args.seed);
save_epoch_folder = '{}/{}'.format('epoch_result', run_name);

try :
    os.mkdir(save_epoch_folder)
except :
    pass

if args.tb_logging :
    from torch.utils.tensorboard import SummaryWriter
    from util.utils import draw_shuffle_array, draw_interp_array, draw_velocity_array, draw_template, draw_img

    tb = SummaryWriter(os.path.join("./runs", run_name));

transformer_file = os.path.join(save_epoch_folder, 'Transformer.pt')
generator_nn_file = os.path.join(save_epoch_folder, 'Generator.pt')
critic_nn_file = os.path.join(save_epoch_folder, "Critic.pt")


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
        args.batch_size, train_size=args.data_num, val_size=10, seed=args.seed, num_classes=5);
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

elif args.exp_name in ["cifar"] : # cifar model
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=128, repeat_num=3,
                              template_dim=template_dim);
    Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, repeat_num=2, template_dim= template_dim);
    Critic_net = CNN_network_bn(in_channel=3, channels=[32, 64, 128, 256], class_num =n_z)

elif args.exp_name in ["fc100"] :  # fc100
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=128, repeat_num=3,
                              template_dim=template_dim);
    Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, repeat_num=2,  repeat_num_v=2, template_dim= template_dim);
    Critic_net = CNN_network_bn(in_channel=3, channels=[32, 64, 128, 256], class_num =n_z)


elif args.exp_name in ["cub200"] :
    args.clf_lr = 0.001
    Generator_net = Generator(n_z, channel_dim=256, repeat_num=3,
                              template_dim=template_dim, img_size=84);
    Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, img_size=84, repeat_num=2, repeat_num_v=2,
                                  template_dim= template_dim);
    Critic_net = CNN_network_bn(in_channel=3, channels=[32, 64, 128, 256], class_num =n_z)



transformer_optimizer = optim.Adam(list(Transformer_net.parameters()),
                       lr=args.lr)
scheduler_transformer = lr_scheduler.StepLR(transformer_optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

generator_optimizer = optim.Adam(list(Generator_net.parameters()) ,
                                lr= args.lr)
scheduler_generator = lr_scheduler.StepLR(generator_optimizer, step_size = args.lr_decay,
                                         gamma= args.gamma);

critic_optimizer = optim.Adam(list(Critic_net.parameters()),
                                lr= args.clf_lr)


criterion = CrossEntropyLoss()
criterion_D = nn.BCELoss()

if args.cuda:
    Transformer_net = Transformer_net.cuda()
    Generator_net = Generator_net.cuda()
    Critic_net = Critic_net.cuda()

def forward_critic(data_loader, eval=False) :

    losses = []
    accs = []

    if eval :
        Critic_net.eval();
    else :
        Critic_net.train();

    for batch_idx, (data, _, label) in enumerate(data_loader):

        data = data.contiguous()

        if args.cuda:
            data, label \
                = data.cuda(), label.cuda()

        data, label = Variable(data), Variable(label)

        out_rec = Critic_net(data);
        probs = F.softmax(out_rec,1);
        prediction = torch.argmax(probs, dim=1);
        correct = (prediction == label).type(torch.float);

        loss_recog = criterion(out_rec, label);

        if not eval :
            critic_optimizer.zero_grad();
            loss_recog.backward();
            critic_optimizer.step();


        losses.append(loss_recog.item());
        accs.append(correct.mean().item())

    return np.mean(losses), np.mean(accs);


def forward_transf(data_loader, losses, eval=False) :

    Critic_net.eval()

    if eval :
        Transformer_net.eval()
        Generator_net.eval();
    else :
        Transformer_net.train()
        Generator_net.train();

    losses["transform_loss"] = []
    losses["critic_loss"] = []
    losses["attention_loss"] = []

    for batch_idx, (data, data2, label) in enumerate(data_loader):

        data, data2 = data.contiguous(), data2.contiguous();


        if args.cuda:
            data, data2, label = data.cuda(), data2.cuda(), label.cuda()

        data, data2, label = Variable(data), Variable(data2), Variable(label)


        # Template
        template = Generator_net(label);

        sp_interv, val_interv = np.random.random(), np.random.random();

        # Transformation (spatial transform applied)
        # transform (template to spatial_transformed)
        spatial_transformed, pixel_offset = Transformer_net.forward_spatial(template, data, label);
        #spatial_transformed2, _ = Transformer_net.forward_spatial(template, data2, label);

        # Transformation (value transformation applied)
        # transform (spatial transformed to final_transformed)
        final_transformed, cls_palette, palette_offset = Transformer_net.forward_value(spatial_transformed,
                                                                                       data,
                                                                                       spatial_transformed.detach(),
                                                                                       label);

        template_img = Transformer_net.colorize_template(template, label);

        with torch.no_grad() :
            spatial_transformed2, _ = Transformer_net.forward_spatial(template, data2, label);


        attention_losses = [];


        for t_i in range(template_dim) :
            pos_sim = 0
            neg_sim = 0;

            for t_j in range(template_dim) :

                attention_data = spatial_transformed[:,t_i].unsqueeze(1)*data;
                attention_data2 = spatial_transformed2[:, t_j].unsqueeze(1) * data2;

                _ = Critic_net(attention_data);
                attention_feature = Critic_net.fin_feature;

                _ = Critic_net(attention_data2);
                attention_feature2 = Critic_net.fin_feature

                if t_i == t_j :
                    pos_sim += cosine_sim(attention_feature, attention_feature2).exp();
                else :
                    neg_sim += cosine_sim(attention_feature, attention_feature2).exp();

                #attention_loss += cosine_dist(attention_feature, attention_feature2, is_normed=False);
                #attention_loss += F.mse_loss(attention_feature, attention_feature2);
                #attention_loss += nll_gaussian(attention_feature, attention_feature2, weighting=1.0, variance=0.1)
                #attention_loss += nll_gaussian(attention_feature, attention_feature2, weighting=1.0, variance=0.1);

            attention_loss = -torch.log(pos_sim/(pos_sim+neg_sim + 1e-6));
            attention_losses.append(attention_loss);

            break;

        out_rec = Critic_net(template_img);
        loss_recog = criterion(out_rec, label);

        attention_loss = torch.stack(attention_losses).mean(0);

        transform_loss = nll_gaussian(final_transformed, data, weighting=1.0, variance=0.1);

        #total_loss = transform_loss + torch.clamp(loss_recog-0.2, min=0.0)
        total_loss = transform_loss + attention_loss#+ loss_recog #+ attention_loss

        if not eval :
            transformer_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            total_loss.backward();

            transformer_optimizer.step();
            generator_optimizer.step()

        losses["transform_loss"].append(transform_loss.detach().cpu().numpy());
        losses["critic_loss"].append(loss_recog.detach().cpu().numpy());
        losses["attention_loss"].append(attention_loss.detach().cpu().numpy())

    scheduler_transformer.step()
    scheduler_generator.step();

    for key in losses :
        if type(losses[key]) == list :
            losses[key] = np.mean(losses[key])



def forward_shuffle(data_loader, figures) :
    Critic_net.eval()
    Transformer_net.eval()
    Generator_net.eval();


    for batch_idx, (data,_, label) in enumerate(data_loader):

        data = data.contiguous()

        if args.cuda:
            data, label = data.cuda(), label.cuda()

        data, label = Variable(data), Variable(label)

        _ = Critic_net(data);

        idxes = torch.where(label == label[0])[0];
        sequence_len = min(len(idxes), 3);

        total_transformed_list = list();
        total_spatial_transformed_list = list();

        template = Generator_net(label[0].unsqueeze(0));

        for src_i in range(sequence_len) :
            row_transformed_list = list();
            row_spatial_transformed_list = list();

            for tar_i in range(sequence_len) :
                src_dat = data[idxes[src_i]];
                tar_dat = data[idxes[tar_i]];

                spatial_transformed, pixel_offset = Transformer_net.forward_spatial(template, src_dat.unsqueeze(0),
                                                                                    label[0].unsqueeze(0));

                spatial_transformed2, _ = Transformer_net.forward_spatial(template, tar_dat.unsqueeze(0),
                                                                                    label[0].unsqueeze(0));


                final_transformed, cls_palette, palette_offset  = Transformer_net.forward_value(spatial_transformed,
                                                                                               tar_dat.unsqueeze(0),
                                                                                                spatial_transformed2,
                                                                                               label[0].unsqueeze(0));

                row_transformed_list.append(final_transformed[0]);
                row_spatial_transformed_list.append(spatial_transformed[0]);

            total_transformed_list.append(torch.stack(row_transformed_list));
            total_spatial_transformed_list.append(torch.stack(row_spatial_transformed_list));


        figure_template = draw_template(template[0], template_dim = template_dim)
        figures.append(figure_template)
        '''
        for t_i in range(template_dim) :
            figure_shuffle_temp = draw_shuffle_array(torch.clamp(template[0, t_i], min=0.0, max=1.0),
                                                data[idxes[:sequence_len]])
            figures.append(figure_shuffle_temp)
        '''

        figure_shuffle = draw_shuffle_array(torch.clamp(torch.stack(total_transformed_list), min=0.0, max=1.0),
                           data[idxes[:sequence_len]])
        figures.append(figure_shuffle)

        '''
        figure_shuffle_tr = draw_shuffle_array(
            torch.clamp(torch.stack(total_spatial_transformed_list), min=0.0, max=1.0),
            data[idxes[:sequence_len]])
        figures.append(figure_shuffle_tr)
        '''

        for t_i in range(template_dim) :
            figure_shuffle_tr = draw_shuffle_array(torch.clamp(torch.stack(total_spatial_transformed_list)[:,:,t_i], min=0.0, max=1.0),
                                                data[idxes[:sequence_len]])
            figures.append(figure_shuffle_tr)


        return;


def forward_interp(data_loader, figures) :
    Critic_net.eval()
    Transformer_net.eval()
    Generator_net.eval();

    for batch_idx, (data, _, label) in enumerate(data_loader):

        data = data.contiguous()

        if args.cuda:
            data, label = data.cuda(), label.cuda()

        data, label = Variable(data), Variable(label)

        idxes = torch.where(label == label[0])[0];
        sequence_len = min(len(idxes), 3);

        tot_spat_interp_list = list();

        tot_appr_interp_list = list();

        template = Generator_net(label[0].unsqueeze(0));

        for tar_i in range(sequence_len):
            tar_dat = data[idxes[tar_i]];

            spat_interp_list = list();

            appr_interp_list = list();
            interp_vals = np.linspace(start=0.0, stop=1.0, num=10)

            for iv in interp_vals:

                spatial_transformed, pixel_offset = Transformer_net.forward_spatial(template,
                                                                                             tar_dat.unsqueeze(0),
                                                                                             label[0].unsqueeze(0),
                                                                                             spatial_interp=iv);
                final_transformed, cls_palette, palette_offset = Transformer_net.forward_value(spatial_transformed,
                                                                                               tar_dat.unsqueeze(0),
                                                                                               spatial_transformed,
                                                                                               label[0].unsqueeze(0));


                spat_interp_list.append(final_transformed[0]);

                spatial_transformed,  pixel_offset = Transformer_net.forward_spatial(template,
                                                                                             tar_dat.unsqueeze(0),
                                                                                             label[0].unsqueeze(0));
                final_transformed, cls_palette, palette_offset = Transformer_net.forward_value(spatial_transformed,
                                                                                               tar_dat.unsqueeze(0),
                                                                                               spatial_transformed,
                                                                                               label[0].unsqueeze(0),
                                                                                               value_interp=iv);

                appr_interp_list.append(final_transformed[0]);

            tot_spat_interp_list.append(torch.stack(spat_interp_list));
            tot_appr_interp_list.append(torch.stack(appr_interp_list));


        fig_sp_velo = draw_velocity_array(pixel_offset.permute(0,3,1,2),
                                          data[idxes[:sequence_len]][0].unsqueeze(0),
                                          tar_dat.unsqueeze(0));

        figures.append(fig_sp_velo)
        fig_sp_interp = draw_interp_array(torch.clamp(torch.stack(tot_spat_interp_list), min=0.0, max=1.0),
                                            data[idxes[:sequence_len]])
        figures.append(fig_sp_interp)

        fig_appr_interp = draw_interp_array(torch.clamp(torch.stack(tot_appr_interp_list), min=0.0, max=1.0),
                                          data[idxes[:sequence_len]])
        figures.append(fig_appr_interp)

# Train model
t_total = time.time()
best_train_loss = np.inf
best_epoch = 0

pbar_clf = tqdm(range(args.epochs_clf));

for epoch in pbar_clf :

    recog_train, acc_train = forward_critic(train_loader, eval=False);
    recog_val, acc_val = forward_critic(valid_loader, eval=True);

    pbar_clf.set_description("acc_val : {}".format(acc_val));
    if args.tb_logging :

        tb.add_scalar("critic_acc_train", acc_train, global_step=epoch)
        tb.add_scalar("critic_acc_val", acc_val, global_step=epoch)

torch.save(Critic_net.state_dict(), critic_nn_file)

t_total = time.time()
best_train_loss = np.inf
best_epoch = 0
recon_losses = [];

for epoch in tqdm(range(args.epochs_gen)):

    first_visit = False;
    t = time.time()
    losses = {}

    forward_transf(train_loader_no_aug, losses, eval=False);
    recon_losses.append(losses["transform_loss"]);
    #recon_val, critic_val = forward_transf(valid_loader, eval=True);
    if args.tb_logging :
        tb.add_scalar("recon_train", losses["transform_loss"], global_step=epoch)
        tb.add_scalar("att_train", losses["attention_loss"], global_step=epoch)
        tb.add_scalar("critic_train", losses["critic_loss"], global_step=epoch)

    if args.tb_logging and epoch % args.vis_interval == 0 :

        figures = []
        forward_shuffle(train_loader_no_aug, figures);
        forward_interp(train_loader_no_aug, figures);


        for idx, figure in enumerate(figures) :
            tb.add_image("synthesis_result_{}".format(idx), figure, global_step=epoch, dataformats="HWC")

    if epoch % args.save_interval == 0 :

        if best_train_loss > losses["transform_loss"] + losses["attention_loss"] :

            best_train_loss = losses["transform_loss"] + losses["attention_loss"];

            torch.save(Transformer_net.state_dict(), transformer_file)
            torch.save(Generator_net.state_dict(), generator_nn_file)

np.save(os.path.join(save_epoch_folder, "recon_loss.npy"), np.stack(recon_losses))
print("Optimization Finished!")




