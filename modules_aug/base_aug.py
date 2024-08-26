

from torchvision import transforms
import torch
import numpy as np
from sklearn.mixture import GaussianMixture

class color_aug(object) :

    def __init__(self):
        self.transform = transforms.Grayscale(num_output_channels=3);
        self.meta_aug=False

    def fit(self, data, label):
        pass


    def __call__(self, img1, idx, idx2) : #

        gray_img = self.transform(img1);

        random_interp = np.random.uniform(low=0.8, high=1.0)

        output_img = gray_img*random_interp + img1*(1-random_interp);

        return output_img

class rotate_aug(object) :

    def __init__(self, rot_range):
        self.min_range = rot_range[0];
        self.max_range = rot_range[1];
        self.transform = transforms.RandomRotation(degrees=(self.min_range, self.max_range));

        self.meta_aug = False

    def fit(self, data, label):
        pass

    def __call__(self, img1, idx) :

        rot_img = self.transform(img1);

        return rot_img

class rot_sca_ela_aug(object) :

    def __init__(self, ):
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.RandomAffine(degrees=0,translate=(0,0),scale=(0.8,1.2))
            ])


        self.meta_aug = False

    def fit(self, data, label):
        pass

    def __call__(self, img1, idx) :

        output_img = self.transform(img1);


        return output_img


class palette_aug(object) :

    def __init__(self, num_k=8):
        self.num_k = num_k;
        self.soft_maps = None;
        self.palettes = None;

        self.meta_aug = False

    def fit(self, data, label) :

        np_data = data.cpu().data.numpy();

        soft_maps = [];
        hard_maps = []
        palletes = [];

        for dat in np_data :

            C, W, H = dat.shape;

            flat_dat = np.reshape(dat, (3, -1)).transpose();
            mean_f = flat_dat.mean(0);
            std_f = flat_dat.std(0);

            model = GaussianMixture(n_components=self.num_k);
            #whiten_dat = (flat_dat - mean_f)/std_f;

            gmm = model.fit(flat_dat)

            cluster_centers = gmm.means_;
            labels_ = gmm.predict(flat_dat);

            soft_labels = gmm.predict_proba(flat_dat);

            hard_labels = np.zeros_like(soft_labels);
            for idx, m in enumerate(labels_) :
                hard_labels[idx][m] = 1.0;

            soft_map = np.reshape(soft_labels.transpose(), (self.num_k, W,H));
            hard_map = np.reshape(hard_labels.transpose(), (self.num_k, W, H));

            soft_maps.append(soft_map);
            hard_maps.append(hard_map);
            palletes.append(cluster_centers);

        self.soft_maps = np.stack(soft_maps)
        self.hard_maps = np.stack(hard_maps)
        self.palettes = np.stack(palletes);


    def palette_to_img(self, palette, soft_map):

        return (palette[:, :, np.newaxis, np.newaxis] * np.expand_dims(soft_map,1)).sum(0)


    def __call__(self, img1, idx, idx2) : #

        output_img = self.palette_to_img(self.palettes[idx],
                                         self.soft_maps[idx]);

        #import cv2

        #cv2.imshow("test", np.transpose(output_img, (1,2,0)));
        #cv2.waitKey(0)
        return torch.FloatTensor(output_img)
