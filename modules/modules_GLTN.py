import torch.nn as nn

from util.utils import gen_onehot_tensor
import torch.nn.functional as F
from torch_deform_conv.deform_conv import th_batch_map_offsets, th_generate_grid
from modules.layers import ToRGB
import numpy as np
import torch


_EPS = 1e-10

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SegmNet(nn.Module):

    def __init__(self, segm_dim, type):
        super(SegmNet, self).__init__()
        self.type = type;

        if type == "cifar_conv":
            layers = []

            # Up Sampling
            layers.append(nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.LeakyReLU())

            layers.append(nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=1, bias=False))
            layers.append(nn.LeakyReLU())

            layers.append(nn.Conv2d(256, segm_dim, kernel_size=4, stride=1, padding=1, bias=False))

        self.segm_net = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_x):

        segm = self.segm_net(input_x);

        segm = F.interpolate(segm, size=input_x.shape[-1],
                             mode='bilinear', align_corners=True)  # *inputs[:,0,:];

        segm = F.softmax(segm * 2, dim=1);

        return segm


class Generator(nn.Module):

    def __init__(self, n_z, channel_dim=64, repeat_num=2, img_size=32, template_dim=3, act_fn="lrelu"):
        super(Generator, self).__init__()
        self.template_dim = template_dim;
        self.n_z = n_z;
        self.repeat_num = repeat_num;
        self.img_size = img_size;

        if act_fn == "relu":
            self.act_func = nn.ReLU()
        elif act_fn == "elu":
            self.act_func = nn.ELU()
        elif act_fn == "lrelu":
            self.act_func = nn.LeakyReLU()


        fc_layers = [];
        if self.img_size ==32 :
            fc_layers.append(nn.Linear(n_z, n_z))
            fc_layers.append(nn.LeakyReLU())
        elif self.img_size == 84:
            fc_layers.append(nn.Linear(n_z, 1024))
            fc_layers.append(nn.LeakyReLU())

        self.fc_layers = nn.Sequential(*fc_layers);

        layers = [];
        if self.img_size == 32 :
            layers.append(nn.ConvTranspose2d(n_z, channel_dim, kernel_size=6, stride=1, padding=0))

        else :
            layers.append(nn.ConvTranspose2d(64, channel_dim, kernel_size=5, stride=2, padding=0))

        layers.append(self.act_func)

        in_channel = channel_dim;

        if self.img_size == 32 :

            for i in range(self.repeat_num):
                out_channel = in_channel // 2;
                layers.append(nn.Conv2d(in_channels=in_channel,
                                                 out_channels=out_channel, kernel_size=4, stride=1, padding=1))
                #layers.append(nn.BatchNorm2d(out_channel))
                layers.append(self.act_func)
                #layers.append(nn.Dropout(0.2))
                layers.append(nn.Upsample(scale_factor=2));

                in_channel = out_channel;

            layers.append(nn.Conv2d(in_channels=in_channel,
                                                 out_channels=self.template_dim, kernel_size=5, padding=1, stride=1))

        elif self.img_size == 84 :
            for i in range(self.repeat_num):
                out_channel = in_channel // 2;
                layers.append(nn.Conv2d(in_channels=in_channel,
                                        out_channels=out_channel, kernel_size=4, stride=1, padding=1))
                #layers.append(nn.ConvTranspose2d(in_channels=in_channel,
                #                        out_channels=out_channel, kernel_size=3, stride=2, padding=1))
                # layers.append(nn.BatchNorm2d(out_channel))
                layers.append(self.act_func)
                # layers.append(nn.Dropout(0.2))
                layers.append(nn.Upsample(scale_factor=2));

                in_channel = out_channel;

            layers.append(nn.Conv2d(in_channels=in_channel,
                                    out_channels=template_dim, kernel_size=7, padding=1, stride=1))

        self.generator = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def shift_image(self, X, dx, dy):
        X = np.roll(X, dy, axis=0)
        X = np.roll(X, dx, axis=1)
        if dy > 0:
            X[:dy, :] = 0
        elif dy < 0:
            X[dy:, :] = 0
        if dx > 0:
            X[:, :dx] = 0
        elif dx < 0:
            X[:, dx:] = 0
        return X

    def gkern(self, l=5, sig=1.):
        """\
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)

    def forward(self, label):


        z_label = gen_onehot_tensor(label.shape[0], label, tot_class=self.n_z, gpu="cuda" == label.device.type);

        if self.img_size == 84 :
            w_label = self.fc_layers(z_label);
            w_label = w_label.reshape(label.shape[0], 64, 4, 4);

        else :
            w_label = self.fc_layers(z_label);
            w_label = w_label.unsqueeze(-1).unsqueeze(-1);


        gen_x = self.generator(w_label);
        gen_x = F.interpolate(gen_x, size=(self.img_size, self.img_size),
                              mode='bilinear',align_corners=True)
        #gen_x = F.softmax(gen_x, dim=1)
        #gen_x = F.sigmoid(gen_x);
        #gen_x = F.gumbel_softmax(gen_x, dim=1)
        #gen_x = F.tanh(gen_x)


        gen_x = torch.clamp(gen_x, min=0.0, max=1.0);

        return gen_x


class Transformer(nn.Module):

    def __init__(self, n_z, in_channel=3, channel_dim=64, repeat_num=2, repeat_num_v=2, img_size=32, template_dim=3, act_fn="lrelu"):
        super(Transformer, self).__init__()
        self.n_z = n_z;
        self.repeat_num = repeat_num
        self.repeat_num_v = repeat_num_v
        self.template_dim = template_dim
        self.in_channel = in_channel
        self.img_size = img_size;

        self._grid_param = None

        if act_fn == "relu":
            self.act_func = nn.ReLU()
        elif act_fn == "elu":
            self.act_func = nn.ELU()
        elif act_fn == "lrelu":
            self.act_func = nn.LeakyReLU()

        #####
        ###   define TR S
        #####
        layer_enc = []
        layer_enc.append(nn.Conv2d(in_channel + 1, channel_dim, kernel_size=3,
                                   stride=1, padding=1))
        layer_enc.append(self.act_func)

        in_channel = channel_dim;

        for i in range(self.repeat_num):
            out_channel = in_channel * 2;
            layer_enc.append(nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2,
                                       padding=1))
            layer_enc.append(self.act_func)

            in_channel = out_channel

        #self.encoder_s = nn.Sequential(*layer_enc);
        self.encoder_s = nn.ModuleList([nn.Sequential(*layer_enc)
                                        for _ in range(template_dim)])

        layer_tr_s = []

        for i in range(self.repeat_num):
            out_channel = in_channel // 2
            layer_tr_s.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4,
                                                 stride=2, padding=1))
            layer_tr_s.append(self.act_func)
            layer_tr_s.append(nn.Dropout(0.2))

            '''
            layer_tr_s.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3,
                                                 stride=1, padding=1))
            layer_tr_s.append(self.act_func)
            layer_tr_s.append(nn.Upsample(scale_factor=2))
            '''
            in_channel = out_channel


        layer_tr_s.append(nn.ConvTranspose2d(in_channel, in_channel, kernel_size=4,
                                             stride=1, padding=0))
        layer_tr_s.append(self.act_func)
        layer_tr_s.append(nn.Dropout(0.2))

        '''
        layer_tr_s.append(nn.ConvTranspose2d(in_channel, in_channel, kernel_size=3,
                                             stride=1, padding=0))
        layer_tr_s.append(self.act_func)
        '''
        layer_tr_s.append(nn.Conv2d(in_channel, 2, kernel_size=4,
                                             stride=1, padding=0))

        #self.transform_s = nn.Sequential(*layer_tr_s)
        self.transform_s = nn.ModuleList([nn.Sequential(*layer_tr_s)
                                        for _ in range(template_dim)])

        #####
        ###   define TR V  (encoder_v)
        #####
        layer_enc = []
        layer_enc.append(nn.Conv2d(self.in_channel, channel_dim, kernel_size=3,
                                   stride=2, padding=0))
        layer_enc.append(self.act_func)

        in_channel = channel_dim;

        for i in range(self.repeat_num_v):
            out_channel = in_channel // 2;
            layer_enc.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2,
                                       padding=0))
            layer_enc.append(self.act_func)
            in_channel = out_channel

        layer_enc.append(nn.Conv2d(in_channel, 32, kernel_size=1, stride=1,
                                   padding=0))
        layer_enc.append(self.act_func)
        layer_enc.append(nn.AdaptiveAvgPool2d(1))
        self.encoder_v = nn.ModuleList([nn.Sequential(*layer_enc) for _ in range(self.template_dim)]);

        self.cls_palette = nn.ModuleList([nn.Sequential(
            nn.Linear(self.n_z, self.n_z),
            nn.LeakyReLU(),
            nn.Linear(self.n_z, self.in_channel),
            nn.Sigmoid(),
        ) for _ in range(self.template_dim)])

        self.cls_offset = nn.ModuleList([nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, self.in_channel),
            nn.Tanh(),
        ) for _ in range(self.template_dim)])

        '''
        self.cls_offset = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, self.in_channel*template_dim),
            nn.Tanh(),
        )'''

        self.appearance_class = nn.ModuleList([ToRGB(1, style_dim=self.n_z) for _ in range(self.template_dim)]);
        self.appearance_offset = nn.ModuleList([ToRGB(1, style_dim=32) for _ in range(self.template_dim)]);

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.)

            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, 2c, h, w) -> (b*c, h, w, 2)"""
        x = x.permute(0, 2, 3, 1);
        # x = x.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]), 2)
        return x

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

    @staticmethod
    def _get_grid(self, x):
        batch_size, input_height, input_width = x.size(0), x.size(2), x.size(3)
        dtype, cuda = x.data.type(), x.data.is_cuda
        if self._grid_param == (batch_size, input_height, input_width, dtype, cuda):
            return self._grid
        self._grid_param = (batch_size, input_height, input_width, dtype, cuda)
        self._grid = th_generate_grid(batch_size, input_height, input_width, dtype, cuda)
        return self._grid

    def generate_with_interpolation_tp(self, inputs, transform_parameter_list):
        output_list = list();

        value_offset_list = list();
        pixel_offset_list = list();

        for transform_parameter in transform_parameter_list:
            output = self.forward(inputs, transform_parameter);
            output_list.append(output[1][:, 1:]);
            # value_offset_list.append(output[2][:,1:]);
            # pixel_offset_list.append(output[3][:,1:]);

        return output_list, output[2][:, 1:], output[3][:, 1:];

    def generate_with_interpolation(self, inputs, rel_rec, rel_send, transform_parameter_list, iter_num, sequence_num):

        output_list = list();
        spatial_transformed_list = list();

        pixel_offset_list = list();

        for transform_parameter in transform_parameter_list:
            output = self.forward(inputs, rel_rec, rel_send, transform_parameter, iter_num, sequence_num);
            output_list.append(output[0]);
            spatial_transformed_list.append(output[2])
            pixel_offset_list.append(output[3])
        return torch.stack(output_list, 1), torch.stack(spatial_transformed_list, 1), torch.stack(pixel_offset_list, 1);

    def transform_with_pixel_offset(self, input, offset):
        # offset = offset*((torch.abs(offset)>1).cpu().type(torch.FloatTensor).cuda()*1.3);
        # x_offset: (b, c, h, w)
        x_offset, grid = th_batch_map_offsets(input, offset, grid=self._get_grid(self, input))

        transformed_result = self._to_b_c_h_w(x_offset, input.shape)

        return transformed_result

    def transform_with_palette(self, cls_colors, offset_colors, sp_transformed):

        class_color_map = (cls_colors[:, :, :, None, None] * sp_transformed.unsqueeze(2)).sum(1);

        offset_color_map = (offset_colors[:, :, :, None, None] * sp_transformed.unsqueeze(2)).sum(1);

        final_transformed = class_color_map + offset_color_map;

        return final_transformed

    def spatial_transform(self, tr_vector, transform_target, spatial_interp=1.0):

        transformer_result = self.transform_s(tr_vector)

        transformer_result = F.interpolate(transformer_result, size=transform_target.shape[2],
                                           mode='bilinear', align_corners=True)  # *inputs[:,0,:];

        # transformer_result = torch.ones_like(transformer_result) * 5;

        offset = transformer_result.permute(0, 2, 3, 1);
        offset = spatial_interp * offset;
        # offset = offset*((torch.abs(offset)>1).cpu().type(torch.FloatTensor).cuda()*1.3);
        # x_offset: (b, c, h, w)
        x_offset, _ = th_batch_map_offsets(transform_target, offset, grid=self._get_grid(self, transform_target))

        sp_transformed = self._to_b_c_h_w(x_offset, transform_target.shape)

        return sp_transformed, offset

    def colorize_template(self, template, label):
        '''
        z_label = gen_onehot_tensor(label.shape[0], label, tot_class=self.n_z, gpu="cuda" == label.device.type);

        class_colors = self.cls_palette(z_label).reshape(label.shape[0],
                                                         self.template_dim,
                                                         self.in_channel);

        class_color_map = (class_colors[:, :, :, None, None] * template.unsqueeze(2)).sum(1);
        '''

        class_colors = [];
        z_label = gen_onehot_tensor(label.shape[0], label, tot_class=self.n_z, gpu="cuda" == label.device.type);

        for j in range(self.template_dim):
            class_color = torch.clamp(self.cls_palette[j](z_label), min=0.0, max=1.0);
            class_colors.append(class_color);

        class_colors = torch.stack(class_colors, 1);
        class_color_map = (class_colors[:, :, :, None, None] * template.unsqueeze(2)).sum(1);

        return class_color_map

    def forward_spatial(self, template, target_data, label, spatial_interp=1.0):

        # import cv2

        # cv2.imshow("test", template[0].permute(1,2,0).cpu().data.numpy());

        sp_transforms = []
        for t_i in range(self.template_dim) :

            x_embed = self.encoder_s[t_i](torch.cat([template[:,t_i].unsqueeze(1),
                                                target_data], dim=1));

            transformer_result = self.transform_s[t_i](x_embed)

            transform_target = template[:,t_i].unsqueeze(1);

            transformer_result = F.interpolate(transformer_result, size=transform_target.shape[2],
                                               mode='bilinear', align_corners=True)  # *inputs[:,0,:];

            # transformer_result = torch.ones_like(transformer_result) * 5;

            offset = transformer_result.permute(0, 2, 3, 1);
            offset = spatial_interp * offset;
            # offset = offset*((torch.abs(offset)>1).cpu().type(torch.FloatTensor).cuda()*1.3);
            # x_offset: (b, c, h, w)
            x_offset, _ = th_batch_map_offsets(transform_target, offset, grid=self._get_grid(self, transform_target))

            sp_transformed = self._to_b_c_h_w(x_offset, transform_target.shape)

            sp_transforms.append(sp_transformed[:,0]);

        return torch.stack(sp_transforms,1), offset

        '''
        sp_transformed_list = []
        offset_list = []
        for temp_idx in range(self.template_dim) :

            x_embed = self.encoder_s[temp_idx](torch.cat([template[:,temp_idx].unsqueeze(1),
                                    target_data],dim=1));

            sp_transformed, offset = self.spatial_transform(x_embed,
                                                            template[:,temp_idx].unsqueeze(1),
                                                            spatial_interp=spatial_interp)

            sp_transformed_list.append(sp_transformed);
            offset_list.append(offset);

        return sp_transformed_list, offset_list;
        '''

    def forward_value(self, sp_transformed, data, att_target, label, value_interp=1.0):

        '''
        z_label = gen_onehot_tensor(label.shape[0], label, tot_class=self.n_z, gpu="cuda" == label.device.type);

        class_colors = self.cls_palette(z_label).reshape(label.shape[0],
                                                         self.template_dim,
                                                         self.in_channel);

        class_color_map = (class_colors[:, :, :, None, None] * sp_transformed.unsqueeze(2)).sum(1);

        #offset_encoding = self.encoder_v(torch.cat([data, att_target], dim=1)).squeeze(-1).squeeze(-1);
        offset_encoding = self.encoder_v(data).squeeze(-1).squeeze(-1);

        offset_colors = self.cls_offset(offset_encoding).reshape(label.shape[0],
                                                         self.template_dim,
                                                         self.in_channel);

        offset_color_map = (offset_colors[:, :, :, None, None] * sp_transformed.unsqueeze(2)).sum(1);

        final_transformed = class_color_map + value_interp * offset_color_map;

        return final_transformed, class_color_map, offset_color_map
        '''

        z_label = gen_onehot_tensor(label.shape[0], label, tot_class=self.n_z, gpu="cuda" == label.device.type);

        class_colors = [];

        for j in range(self.template_dim) :
            class_color = torch.clamp(self.cls_palette[j](z_label), min=0.0, max=1.0);
            class_colors.append(class_color);

        class_colors =  torch.stack(class_colors,1);
        class_color_map = (class_colors[:, :, :, None, None] * sp_transformed.unsqueeze(2)).sum(1);

        offset_colors= [];
        for j in range(self.template_dim) :
            #offset_encoding = self.encoder_v(torch.cat([sp_transformed[:,j], att_target[:,j].unsqueeze(1) * data], dim=1))\
            #    .squeeze(-1).squeeze(-1);

            offset_encoding = self.encoder_v[j](data).squeeze(-1).squeeze(-1);
            offset = self.cls_offset[j](offset_encoding);
            offset_colors.append(offset)

        offset_colors = torch.stack(offset_colors,1);
        offset_color_map = (offset_colors[:, :, :, None, None] * sp_transformed.unsqueeze(2)).sum(1);

        final_transformed = class_color_map + value_interp*offset_color_map;

        return final_transformed, class_colors, offset_colors

        '''

        z_label = gen_onehot_tensor(label.shape[0], label, tot_class=self.n_z, gpu="cuda" == label.device.type);

        class_color_maps = [];
        for j in range(self.template_dim) :
            class_color_map = self.appearance_class[j](sp_transformed[:,j].unsqueeze(1), z_label);
            class_color_maps.append(class_color_map);

        offset_color_maps = [];
        embed_v = self.encoder_v(data).squeeze(-1).squeeze(-1);
        for j in range(self.template_dim) :
            offset_color_map = self.appearance_offset[j](sp_transformed[:,j].unsqueeze(1), embed_v);
            offset_color_maps.append(offset_color_map);

        class_color_maps = torch.stack(class_color_maps).sum(0);
        offset_color_maps = torch.stack(offset_color_maps).sum(0);

        final_transformed = class_color_maps + value_interp*offset_color_maps

        return final_transformed, class_color_maps, offset_color_maps
        '''