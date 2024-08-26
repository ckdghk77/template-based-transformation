
import numpy as np;
import os
import torch
class synthe_aug(object) :
    def __init__(self, aug_folder="./out"):

        self.load_folder = aug_folder;

        self.label_num = 0;
        self.data_num = 0;

        self.meta_aug = False

    def fit(self, data, label):

        cs = np.unique(label.cpu().data.numpy());

        self.label_num = len(cs);
        self.data_num = len(data)

        self.datas = []
        for c in cs :

            np_img = np.load(os.path.join(self.load_folder, "{}.npy".format(c)));

            self.datas.append(torch.FloatTensor(np_img))

    def __call__(self, img1, index, index2):  #
        label_idx = index//self.label_num;

        candidates = self.datas[label_idx];

        idx = np.random.choice(len(candidates));

        return candidates[idx]

