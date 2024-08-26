

import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_res20(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_res20, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channel=3):
        super(ResNet_20, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        self.out1 = out.clone()
        out = self.layer2(out)
        self.out2 = out.clone()
        out = self.layer3(out)
        self.out3 = out.clone()
        self.cam_filter = self.out3.clone();

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        self.out_fin = out.clone()
        out = self.linear(out)
        return out

    def get_cam_activation(self, ):

        output = self.linear(self.out_fin);

        pred_label = torch.argmax(output,dim=1);

        weight_t = self.linear.weight.data
        target_weight = weight_t[pred_label,:];

        cam_outs = [];

        for weight, filter in zip(target_weight, self.cam_filter) :
            cam_out = (weight.unsqueeze(-1).unsqueeze(-1) * filter).sum(0)

            norm_cam_out = (cam_out - cam_out.min()) / (cam_out.max() - cam_out.min() + 1e-8);

            cam_outs.append(norm_cam_out);

        return cam_outs

    def get_activations(self, layers):

        outputs = []

        for layer in layers  :
            if layer == 1 :
                output = self.out3
            elif layer == 2 :
                output = self.out3
            elif layer == 3:
                output = self.out3
            elif layer == 4:
                output = self.out3
            elif layer == 5 :
                output = self.fin_feature
            else :
                raise Exception("Layer Error")

            outputs.append(output)
        return outputs

    def get_layer_activation(self, layer_no) :
        if layer_no == 1 :
            return self.out1;
        elif layer_no == 2 :
            return self.out2
        elif layer_no == 3 :
            return self.out3

    def forward_from(self, feature, layer_no=3):
        if layer_no <= 1 :
            feature = self.layer2(feature)
        if layer_no <= 2 :
            feature= self.layer3(feature);
        if layer_no <= 3 :
            feature = F.avg_pool2d(feature, feature.size()[3])
            feature = feature.view(feature.size(0), -1)
            out = self.linear(feature)

        return out;

def resnet20(in_channel=1, class_num=10):
    return ResNet_20(BasicBlock_res20, [3,3,3],in_channel=in_channel, num_classes = class_num)
