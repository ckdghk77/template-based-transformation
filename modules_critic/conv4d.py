
import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS = 1e-10

class CNN_network_bn(nn.Module):
    def __init__(self, in_channel, channels=[8, 16, 32, 64], class_num=10, clf_type=0):
        super(CNN_network_bn, self).__init__()

        if clf_type == 0 :

            self.ds_feature = nn.Sequential(
                nn.Conv2d(in_channel, channels[0], 3, 1, 1),
                nn.BatchNorm2d(channels[0]),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(channels[0], channels[1], 3, 1, 1),
                nn.BatchNorm2d(channels[1]),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(channels[1], channels[2], 3, 1, 1),
                nn.BatchNorm2d(channels[2]),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(channels[2], channels[3], 3, 1, 0),
                nn.BatchNorm2d(channels[3]),
                nn.LeakyReLU()
            )
        elif clf_type == 1 :
            self.ds_feature = nn.Sequential(
                nn.Conv2d(in_channel, channels[0], 3, 1, 1),
                nn.BatchNorm2d(channels[0]),
                nn.LeakyReLU(),
                nn.AvgPool2d(2),

                nn.Conv2d(channels[0], channels[1], 3, 1, 1),
                nn.BatchNorm2d(channels[1]),
                nn.LeakyReLU(),
                nn.AvgPool2d(2),

                nn.Conv2d(channels[1], channels[2], 3, 1, 1),
                nn.BatchNorm2d(channels[2]),
                nn.LeakyReLU(),
                nn.AvgPool2d(2),

                nn.Conv2d(channels[2], channels[3], 3, 1, 0),
                nn.BatchNorm2d(channels[3]),
                nn.LeakyReLU()
            )
        elif clf_type == 2 :
            self.ds_feature = nn.Sequential(
                nn.Conv2d(in_channel, channels[0], 3, 2, 1),
                nn.BatchNorm2d(channels[0]),
                nn.LeakyReLU(),

                nn.Conv2d(channels[0], channels[1], 3, 2, 1),
                nn.BatchNorm2d(channels[1]),
                nn.LeakyReLU(),

                nn.Conv2d(channels[1], channels[2], 3, 2, 1),
                nn.BatchNorm2d(channels[2]),
                nn.LeakyReLU(),

                nn.Conv2d(channels[2], channels[3], 3, 1, 0),
                nn.BatchNorm2d(channels[3]),
                nn.LeakyReLU()
            )

        self.fin_feature = None;
        self.fc_out = nn.Linear(channels[3], class_num);
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)


    def get_cam_activation(self,):

        output = self.fc_out(self.fin_feature.squeeze());

        #pred_label = torch.argmax(output,dim=1);
        pred_label = torch.argmax(output,1);

        weight_t = self.fc_out.weight.data
        target_weight = weight_t[pred_label,:];

        cam_outs = [];

        for weight, filter in zip(target_weight, self.conv_feature) :
            cam_out = (weight.unsqueeze(-1).unsqueeze(-1) * filter).sum(0)

            norm_cam_out = (cam_out - cam_out.min()) / (cam_out.max() - cam_out.min() + 1e-8);

            cam_outs.append(norm_cam_out);

        return cam_outs

    def forward(self, input_x):

        feature = self.ds_feature(input_x);
        self.conv_feature = feature.clone()
        feature = F.adaptive_avg_pool2d(feature, (1,1));

        fin_feature = feature.view(feature.shape[0],-1);

        self.fin_feature = fin_feature.clone()

        self.output = self.fc_out(fin_feature).clone()

        return self.output

    def get_activation(self, ) :
        return self.fin_feature