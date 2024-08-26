import torch.nn as nn

from util.utils import *
from util.data_loader import *
import torch.optim as optim
from torch.utils.data import DataLoader


class SSL_network(nn.Module):
    def __init__(self, mlp_nn, data_loader, args, class_num=10):
        super(SSL_network, self).__init__()

        self.class_num = class_num;
        self.epochs = 400;


        self.mlp_nn = mlp_nn
        self.is_gpu = args.cuda



        self.d_loader = data_loader


        for m in self.mlp_nn.modules() :
            if isinstance(m, nn.Linear) :
                for param in m.parameters() :
                    param.requires_grad = False

            if isinstance(m, nn.BatchNorm2d) :
                for param in m.parameters() :
                    param.requires_grad = False


        self.ssl_optimizer = optim.Adam(self.mlp_nn.parameters(),
                                            lr=0.005)

        self.class_criterion = nn.CrossEntropyLoss();


    def train(self,):

        self.mlp_nn.train()

        for e in range(self.epochs):

            for b_idx, (b_input, _, b_label) in enumerate(self.d_loader):


                if self.is_gpu :
                    b_input, b_label = b_input.cuda(), b_label.cuda()
                    #eig_olds = [eig_old.cuda() for eig_old in eig_olds];

                input_data = b_input
                target_labels = gen_one_hot_tensor(b_input.shape[0], b_label,
                                                   tot_class=self.class_num, prob=1.0, gpu=self.is_gpu)


                logits = self.mlp_nn(input_data);

                rec_loss = self.softXExnt(logits, target_labels);
                #rec_loss = self.class_criterion(logits, torch.argmax(target_labels,1));

                if e % 200 == 0 and b_idx == 0:
                    print("ep {:4d} : loss_1 {:2f}".format(e,  rec_loss));

                self.ssl_optimizer.zero_grad()
                #(feature_loss + rec_loss).backward()  # cos + cos works good
                (rec_loss).backward()  # cos + cos works good
                self.ssl_optimizer.step()


    def softXExnt(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]


