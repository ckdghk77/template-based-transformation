from __future__ import division
from __future__ import print_function

import time
import argparse
import datetime

import torch.optim as optim
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

import torch.nn.functional as F

import os
import numpy as np

import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')   ##  59 good
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train classifier')

parser.add_argument('--batch-size', type=int, default=32,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--exp-name', type=str, default="svhn",
                    help='exp-name (e.g "moving_mnist", "svhn", "cifar",'
                         '"cub200", "m_imgnet")')

parser.add_argument('--data-num', type=int, default=10,
                    help='data number')
parser.add_argument('--n-z', type=int, default=10,
                    help='gen class num')

parser.add_argument('--aug-policy', type=str, default="proposed",
                    help="which augmentation policy to use (1. base, 2. color, "
                         "3. palette, 4. rotate, 5. autoaug, 6. proposed, 7. proposed files)")
parser.add_argument('--clf-name', type=str, default="base",
                    help="clf type  (base, wideresnet, pyramid, res20, )")
parser.add_argument('--val-interval', type=int, default=50,
                    help="validate interval (for img)")
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
run_name = '{}_aug_expm_{}_{}_{}_{}/'.format(args.exp_name,  args.clf_name,
                                                      args.aug_policy, args.data_num, args.seed);

save_epoch_folder = '{}/{}/'.format('epoch_result', run_name);

if args.tb_logging :
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join("./runs",run_name));

try :
    os.mkdir(save_epoch_folder)
except :
    pass

if args.aug_policy == "base" :
    aug_model = None;
elif args.aug_policy == "color":
    from modules_aug.base_aug import color_aug
    aug_model = color_aug();
elif args.aug_policy == "palette" :
    from modules_aug.base_aug import palette_aug
    aug_model = palette_aug();
elif args.aug_policy == "rotate" :
    from modules_aug.base_aug import rotate_aug
    aug_model = rotate_aug((-20,20));
elif args.aug_policy == "rot_sca_ela" :
    from modules_aug.base_aug import rot_sca_ela_aug
    aug_model = rot_sca_ela_aug();
elif args.aug_policy == "autoaug" :
    from modules_aug.meta_aug import autoaug
    aug_model = autoaug(args.exp_name)
elif args.aug_policy == "randaug" :
    from modules_aug.meta_aug import randaug
    aug_model = randaug(args.exp_name)
elif args.aug_policy == "trivialaug":
    from modules_aug.meta_aug import trivialaug
    aug_model = trivialaug(args.exp_name)


elif args.aug_policy == "proposed" :
    from modules_aug.proposed_aug import proposed_aug
    aug_model = proposed_aug(exp_name=args.exp_name, data_num=args.data_num, seed=args.seed);
elif args.aug_policy == "proposed_file" :
    from modules_aug.proposed_aug import proposed_aug_file
    aug_model = proposed_aug_file(exp_name=args.exp_name, data_num=args.data_num, seed=args.seed);



# Load Dataset
if args.exp_name == "c_mnist" :  # c_mnist dataset
    from util.data_loader import load_data_c_mnist
    class_num=10
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_c_mnist(
        args.batch_size, train_size=6, val_size=200, seed=args.seed);
elif args.exp_name == "svhn": # svhn dataset
    class_num=10
    from util.data_loader import load_data_svhn
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_svhn(
        args.batch_size, train_size=args.data_num, val_size=1000, seed=args.seed, aug_model=aug_model);
elif args.exp_name == "cifar":  # cifar dataset
    class_num=10
    from util.data_loader import load_data_cifar
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_cifar(
        args.batch_size, train_size=args.data_num, val_size=200, seed=args.seed, aug_model=aug_model);
elif args.exp_name == "fc100" :  # cifar100 dataset
    from util.data_loader import load_data_cifar100
    class_num=5
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_cifar100(
        args.batch_size, train_size=args.data_num, val_size=10, seed=args.seed, num_classes=5, aug_model=aug_model);
elif args.exp_name == "cub200" : # CUB200 dataset
    from util.data_loader import load_data_CUB200
    class_num=5
    train_loader, train_loader_no_aug, valid_loader, train_x, train_y, transform_train = load_data_CUB200(
        args.batch_size, train_size=args.data_num, val_size=20, seed=args.seed, num_classes=5, aug_model=aug_model);

if args.clf_name == "base" :
    from modules_critic.conv4d import CNN_network_bn
    classifier = CNN_network_bn(in_channel=3, channels=[32, 64, 128, 256], class_num = class_num)
elif args.clf_name == "res20" :
    from modules_critic.res20 import resnet20
    classifier = resnet20(in_channel=3, class_num=class_num);
elif args.clf_name == "wideresnet" :
    from modules_critic.wide_resnet import Wide_ResNet
    classifier = Wide_ResNet(28, 10, 0.3, num_classes=class_num);
elif args.clf_name == "pyramid" :
    from modules_critic.pyramid import pyramidnet
    classifier = pyramidnet(num_classes=class_num);

optimizer = optim.Adam(list(classifier.parameters()) ,
                                lr= args.lr)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[400, 800, 1200, 1600], gamma=0.8)

criterion = CrossEntropyLoss()

if args.cuda:
    classifier = classifier.cuda()

def forward(data_loader, eval=False) :

    classifier.eval() if eval else classifier.train()
    accs = []
    ces = []

    for batch_idx, (data, _, label) in enumerate(data_loader):

        data = data.contiguous()

        if args.cuda:
            data, label = data.cuda(), label.cuda()

        data, label = Variable(data), Variable(label)

        logits = classifier(data);
        probs = F.softmax(logits, 1);

        loss = criterion(logits, label);

        if not eval :

            optimizer.zero_grad()
            loss.backward()
            optimizer.step();

        prediction = torch.argmax(probs, dim=1);
        correct = (prediction == label);

        accs.extend(correct.detach().cpu().numpy());
        ces.append(loss.detach().cpu().numpy());

    accs = 100 * np.mean(accs);
    return np.mean(ces), accs

# Train model
t_total = time.time()
best_train_loss = np.inf
best_epoch = 0

progress = tqdm(range(args.epochs));

for epoch in progress :

    ce_train, acc_train = forward(train_loader, eval=False);
    scheduler.step()
    if epoch % args.val_interval == 0 :
        ce_val, acc_val = forward(valid_loader, eval=True);

    progress.set_description("(acc_val = {:.4f})".format(acc_val))
    if epoch % args.val_interval==0 and args.tb_logging :

        tb.add_scalar("ce_train", ce_train, global_step=epoch)
        tb.add_scalar("ce_val", ce_val, global_step=epoch)
        tb.add_scalar("acc_train", acc_train, global_step=epoch)
        tb.add_scalar("acc_val", acc_val, global_step=epoch)

torch.save(classifier.state_dict(),os.path.join(save_epoch_folder, 'clf.pt'));

print("Optimization Finished!")




