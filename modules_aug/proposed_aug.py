import os

from modules.modules_GLTN import Generator, Transformer
import torch
import numpy as np

from torchvision import transforms

class proposed_aug_file(object) :
    def __init__(self, exp_name, data_num, seed, is_gpu=True, root_dir="./epoch_result"):

        self.load_folder = "{}/exp_{}_prop_tr_gen_{}_{}".format(root_dir, exp_name, data_num, seed);


        self.meta_aug = False

    def fit(self, data, label):

        cs = np.unique(label.cpu().data.numpy());

        datas = []
        for c in cs :

            np_img = np.load(os.path.join(self.load_folder, "{}.npy".format(c)));

            datas.append(torch.FloatTensor(np_img))

        return datas;


    def __call__(self, img1, index, index2):  #

        return img1



class proposed_addaug(object) :
    def __init__(self, exp_name, data_num, seed, is_gpu=True, root_dir="./out"):

        self.is_gpu = is_gpu;
        load_folder = "{}/exp_{}_prop_tr_gen_temp_{}_{}".format(root_dir, exp_name, data_num,seed);

        Generator_file = os.path.join(load_folder, "Generator.pt");
        Transformer_file = os.path.join(load_folder, "Transformer.pt");

        # Load Model
        if exp_name in ["c_mnist"] :
            n_z=10
            template_dim=2
            self.Generator_net = Generator(n_z, channel_dim=32, repeat_num=3, template_dim=template_dim);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=32, repeat_num=2, template_dim=template_dim);

        elif exp_name in ["cifar"]:  # cifar model
            n_z = 10;
            template_dim = 3
            self.Generator_net = Generator(n_z, channel_dim=128, repeat_num=3, template_dim=template_dim);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, repeat_num=2, template_dim=template_dim);

        elif exp_name in ["fc100"] :
            n_z = 5;
            template_dim = 5
            self.Generator_net = Generator(n_z, channel_dim=128, repeat_num=3,
                                      template_dim=template_dim);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, repeat_num=2, repeat_num_v=2,
                                          template_dim=template_dim);

        elif exp_name in ["cub200"] :
            n_z = 5;
            template_dim = 5;
            self.Generator_net = Generator(n_z, channel_dim=256, repeat_num=3,
                                      template_dim=template_dim, img_size=84);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, img_size=84, repeat_num=2, repeat_num_v=2,
                                          template_dim=template_dim);


        elif exp_name in ["svhn"] :
            n_z = 10;
            template_dim = 2
            self.Generator_net = Generator(n_z, channel_dim=64, repeat_num=3, template_dim=template_dim);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=64, repeat_num=2, template_dim=template_dim);


        self.Generator_net.load_state_dict(torch.load(Generator_file,map_location=torch.device("cpu")));
        self.Transformer_net.load_state_dict(torch.load(Transformer_file,map_location=torch.device("cpu")));

        self.pixel_offsets = None
        self.cls_palettes = None
        self.palette_offsets = None
        self.templates = None

        self.meta_aug = True

    def fit(self, data, label):

        self.Generator_net.eval();
        self.Transformer_net.eval();

        if self.is_gpu :
            data, label = data.cuda(), label.cuda()
            self.Generator_net.cuda()
            self.Transformer_net.cuda()


        template = self.Generator_net(label);

        spatial_transformed, pixel_offset = self.Transformer_net.forward_spatial(template,
                                                                                  data,
                                                                                  label);

        _, cls_palette, palette_offset = self.Transformer_net.forward_value(spatial_transformed,
                                                                                data,
                                                                                spatial_transformed.detach(),
                                                                                label);

        self.templates = template.cpu().data;
        self.pixel_offsets = pixel_offset.cpu().data;
        self.cls_palettes = cls_palette.cpu().data;
        self.palette_offsets = palette_offset.cpu().data;

        self.Generator_net.cpu()
        self.Transformer_net.cpu()

    def __call__(self, img1, index, index2) : #
        #sv, iv = np.random.uniform(0.7,1.0), np.random.uniform(0.8,1.0)
        sv, iv = np.random.uniform(1.0,1.0), np.random.uniform(0.8,1.0)


        sp_transformed = self.Transformer_net.transform_with_pixel_offset(self.templates[index:index+1],
                                                                          sv*self.pixel_offsets[index:index+1]);

        final_transformed = self.Transformer_net.transform_with_palette(self.cls_palettes[index2:index2+1],
                                                                       iv*self.palette_offsets[index2:index2+1],
                                                                        sp_transformed);

        transform_train = [
            transforms.RandAugment(),
            transforms.ToTensor(),
        ]


        return final_transformed[0]


class proposed_aug(object) :

    def __init__(self, exp_name, data_num, seed, is_gpu=True, root_dir="./epoch_result"):

        self.is_gpu = is_gpu;
        load_folder = "{}/exp_{}_prop_tr_gen_{}_{}".format(root_dir, exp_name, data_num,seed);

        Generator_file = os.path.join(load_folder, "Generator.pt");
        Transformer_file = os.path.join(load_folder, "Transformer.pt");

        # Load Model
        if exp_name in ["c_mnist"] :
            n_z=10
            template_dim=2
            self.Generator_net = Generator(n_z, channel_dim=32, repeat_num=3, template_dim=template_dim);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=32, repeat_num=2, template_dim=template_dim);

        elif exp_name in ["cifar"]:  # cifar model
            n_z = 10;
            template_dim = 3
            self.Generator_net = Generator(n_z, channel_dim=128, repeat_num=3, template_dim=template_dim);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, repeat_num=2, template_dim=template_dim);

        elif exp_name in ["fc100"] :
            n_z = 5;
            template_dim = 5
            self.Generator_net = Generator(n_z, channel_dim=128, repeat_num=3,
                                      template_dim=template_dim);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, repeat_num=2, repeat_num_v=2,
                                          template_dim=template_dim);

        elif exp_name in ["cub200"] :
            n_z = 5;
            template_dim = 5;
            self.Generator_net = Generator(n_z, channel_dim=256, repeat_num=3,
                                      template_dim=template_dim, img_size=84);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=128, img_size=84, repeat_num=2, repeat_num_v=2,
                                          template_dim=template_dim);


        elif exp_name in ["svhn"] :
            n_z = 10;
            template_dim = 2
            self.Generator_net = Generator(n_z, channel_dim=64, repeat_num=3, template_dim=template_dim);
            self.Transformer_net = Transformer(n_z, in_channel=3, channel_dim=64, repeat_num=2, template_dim=template_dim);


        self.Generator_net.load_state_dict(torch.load(Generator_file,map_location=torch.device("cpu")));
        self.Transformer_net.load_state_dict(torch.load(Transformer_file,map_location=torch.device("cpu")));

        self.pixel_offsets = None
        self.cls_palettes = None
        self.palette_offsets = None
        self.templates = None

        self.meta_aug = False

    def fit(self, data, label):

        self.Generator_net.eval();
        self.Transformer_net.eval();

        if self.is_gpu :
            data, label = data.cuda(), label.cuda()
            self.Generator_net.cuda()
            self.Transformer_net.cuda()


        template = self.Generator_net(label);

        spatial_transformed, pixel_offset = self.Transformer_net.forward_spatial(template,
                                                                                  data,
                                                                                  label);

        _, cls_palette, palette_offset = self.Transformer_net.forward_value(spatial_transformed,
                                                                                data,
                                                                                spatial_transformed.detach(),
                                                                                label);

        self.templates = template.cpu().data;
        self.pixel_offsets = pixel_offset.cpu().data;
        self.cls_palettes = cls_palette.cpu().data;
        self.palette_offsets = palette_offset.cpu().data;

        self.Generator_net.cpu()
        self.Transformer_net.cpu()

    def __call__(self, img1, index, index2) : #
        #sv, iv = np.random.uniform(0.7,1.0), np.random.uniform(0.8,1.0)
        sv, iv = np.random.uniform(1.0,1.0), np.random.uniform(0.8,1.0)


        sp_transformed = self.Transformer_net.transform_with_pixel_offset(self.templates[index:index+1],
                                                                          sv*self.pixel_offsets[index:index+1]);

        final_transformed = self.Transformer_net.transform_with_palette(self.cls_palettes[index2:index2+1],
                                                                       iv*self.palette_offsets[index2:index2+1],
                                                                        sp_transformed);


        '''
        import cv2
        cv2.imshow("test", final_transformed[0][(2,1,0),:,:].permute(1,2,0).cpu().data.numpy());
        cv2.waitKey(0)
        '''
        return final_transformed[0]
